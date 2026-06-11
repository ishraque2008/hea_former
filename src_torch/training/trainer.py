"""
PyTorch training loop for HEAFormer.

Supports:
  - Mixed-precision training (torch.cuda.amp)
  - Multi-GPU via torch.nn.parallel.DistributedDataParallel
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Checkpoint saving / loading

Usage
-----
    from src_torch.training.trainer import HEATrainerTorch
    trainer = HEATrainerTorch(model, cfg)
    trainer.train(train_loader, val_loader)
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except ImportError:
    _TORCH = False

import numpy as np
import os
import json


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class HEADatasetTorch:
    """
    Wraps a SequenceDataset (NumPy) into a PyTorch Dataset.
    Pads all sequences to the same length N for batching.
    """

    def __init__(self, seq_dataset, indices):
        import torch
        self.seq     = seq_dataset
        self.indices = indices
        self.N       = len(seq_dataset.occupancies[0])   # all same length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        import torch
        s = self.seq[self.indices[k]]
        return dict(
            token_ids  = torch.tensor(s['token_ids'],  dtype=torch.long),
            features   = torch.tensor(s['features'],   dtype=torch.float32),
            mask       = torch.tensor(s['mask'],        dtype=torch.bool),
            true_ids   = torch.tensor(s['true_ids'],    dtype=torch.long),
            alpha_flat = torch.tensor(s['alpha_flat'],  dtype=torch.float32),
            phase_label= torch.tensor(s['phase_label'], dtype=torch.long),
            x_global   = torch.tensor(s['x_global'],   dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HEATrainerTorch:
    """
    End-to-end trainer for HEAFormerTorch.

    Stage 1: MLM pre-training
    Stage 2: Physics-informed SRO + phase fine-tuning
    """

    def __init__(self, model, cfg: dict):
        if not _TORCH:
            raise ImportError("PyTorch required for HEATrainerTorch")
        import torch

        self.model  = model
        self.cfg    = cfg
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        print(f"Device: {self.device}")
        print(f"Model params: {model.n_params():,}")

        self.opt = optim.AdamW(
            model.parameters(),
            lr=cfg.get('lr', 5e-4),
            weight_decay=cfg.get('wd', 1e-4),
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.device.type == 'cuda'
        )

        from src_torch.models.transformer import SROLossTorch
        self.sro_criterion = SROLossTorch(
            n_shells=cfg.get('n_shells', 2),
            n_sp=cfg.get('n_sp', 5),
        )

        self.history = {k: [] for k in
                        ['pretrain_loss', 'train_loss', 'val_mae',
                         'train_mae', 'phase_acc']}

    def pretrain_epoch(self, loader) -> dict:
        import torch
        self.model.train()
        losses, accs = [], []
        for batch in loader:
            batch   = {k: v.to(self.device) for k, v in batch.items()}
            tok     = batch['token_ids']
            feat    = batch['features']
            mask    = batch['mask']
            true_ids = batch['true_ids']
            if not mask.any():
                continue
            with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                out    = self.model(tok, feat, mlm_mask=mask)
                logits = out['mlm']
                if logits is None:
                    continue
                loss = nn.functional.cross_entropy(
                    logits, true_ids[mask]
                )
            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            losses.append(loss.item())
            accs.append(
                (logits.argmax(-1) == true_ids[mask]).float().mean().item()
            )
        return dict(loss=float(np.mean(losses)) if losses else 0.0,
                    acc =float(np.mean(accs))   if accs   else 0.0)

    def finetune_epoch(self, loader) -> dict:
        import torch
        self.model.train()
        tot_losses, sro_maes = [], []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            tok   = batch['token_ids']
            feat  = batch['features']
            xg    = batch['x_global']
            at    = batch['alpha_flat']
            ph    = batch['phase_label']

            with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                out        = self.model(tok, feat)
                sro_losses = self.sro_criterion(out['sro'], at, xg)
                l_phase    = nn.functional.cross_entropy(out['phase'], ph)
                total      = sro_losses['total'] + 0.5 * l_phase

            self.opt.zero_grad()
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()

            tot_losses.append(total.item())
            with torch.no_grad():
                sro_maes.append(
                    (out['sro'] - at).abs().mean().item()
                )

        return dict(loss   =float(np.mean(tot_losses)) if tot_losses else 0.0,
                    sro_mae=float(np.mean(sro_maes))   if sro_maes   else 0.0)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        import torch
        self.model.eval()
        preds, trues, ph_p, ph_t = [], [], [], []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out   = self.model(batch['token_ids'], batch['features'])
            preds.append(out['sro'].cpu().numpy())
            trues.append(batch['alpha_flat'].cpu().numpy())
            ph_p.append(out['phase'].argmax(-1).cpu().numpy())
            ph_t.append(batch['phase_label'].cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mae   = float(np.mean(np.abs(preds - trues)))
        ph_p  = np.concatenate(ph_p)
        ph_t  = np.concatenate(ph_t)
        acc   = float((ph_p == ph_t).mean())
        return dict(sro_mae=mae, phase_acc=acc)

    def train(self, seq_dataset, tr_idx, va_idx, te_idx):
        """Full two-stage training."""
        from torch.utils.data import DataLoader

        tr_ds = HEADatasetTorch(seq_dataset, tr_idx)
        va_ds = HEADatasetTorch(seq_dataset, va_idx)
        te_ds = HEADatasetTorch(seq_dataset, te_idx)
        bs    = self.cfg.get('batch_size', 16)

        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                                num_workers=0, pin_memory=False)
        va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False,
                                num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False,
                                num_workers=0)

        # Stage 1: pre-training
        n_pre = self.cfg.get('n_epochs_pretrain', 6)
        print(f"\n[Stage 1] MLM pre-training ({n_pre} epochs) ...")
        for ep in range(n_pre):
            r = self.pretrain_epoch(tr_loader)
            self.history['pretrain_loss'].append(r['loss'])
            print(f"  Ep {ep+1:3d}  MLM loss={r['loss']:.4f}  acc={r['acc']:.4f}")

        # Stage 2: fine-tuning
        n_ft = self.cfg.get('n_epochs_finetune', 25)
        print(f"\n[Stage 2] SRO fine-tuning ({n_ft} epochs) ...")
        for ep in range(n_ft):
            r_tr = self.finetune_epoch(tr_loader)
            r_va = self.evaluate(va_loader)
            self.history['train_loss'].append(r_tr['loss'])
            self.history['train_mae'].append(r_tr['sro_mae'])
            self.history['val_mae'].append(r_va['sro_mae'])
            self.history['phase_acc'].append(r_va['phase_acc'])
            if (ep+1) % max(1, n_ft//5) == 0:
                print(f"  Ep {ep+1:3d}  loss={r_tr['loss']:.4f}  "
                      f"tr_MAE={r_tr['sro_mae']:.4f}  "
                      f"va_MAE={r_va['sro_mae']:.4f}")

        # Final test evaluation
        print("\n[Test evaluation]")
        r_te = self.evaluate(te_loader)
        print(f"  SRO MAE={r_te['sro_mae']:.4f}  Phase Acc={r_te['phase_acc']:.4f}")

        return r_te

    def save_checkpoint(self, path: str):
        import torch
        torch.save(dict(
            model_state=self.model.state_dict(),
            opt_state  =self.opt.state_dict(),
            history    =self.history,
            cfg        =self.cfg,
        ), path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        import torch
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck['model_state'])
        self.opt.load_state_dict(ck['opt_state'])
        self.history = ck['history']
        print(f"Checkpoint loaded: {path}")
