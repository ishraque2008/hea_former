"""
Sklearn baseline models for ablation study.

Baselines (all multi-output aware):
  Ridge         -- linear, composition-only features
  CompositionMLP -- MLP, composition-only features
  LocalEnvMLP   -- MLP, aggregated local-env features
  RandomForest  -- RF, aggregated local-env features
  FrozenTrans   -- linear head on frozen encoder pooled features
"""

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, f1_score


def _reg_pipeline(estimator):
    return Pipeline([('sc', StandardScaler()),
                     ('m',  MultiOutputRegressor(estimator, n_jobs=-1))])

def _cls_pipeline(estimator):
    return Pipeline([('sc', StandardScaler()), ('m', estimator)])


def _eval_reg(model, Xte, yte):
    yp = model.predict(Xte)
    return dict(
        mae  = float(mean_absolute_error(yte, yp)),
        rmse = float(np.sqrt(np.mean((yp - yte)**2))),
        r2   = float(r2_score(yte, yp, multioutput='uniform_average')),
    )

def _eval_cls(model, Xte, yte):
    yp = model.predict(Xte)
    return dict(
        acc = float((yp == yte).mean()),
        f1  = float(f1_score(yte, yp, average='macro', zero_division=0)),
    )


def run_all_baselines(seq_dataset, train_idx, test_idx,
                      verbose=True) -> dict:
    """
    Run Ridge / MLP / RF on both SRO regression and phase classification.
    Returns nested dict: {task: {model_name: metric_dict}}
    """
    # ---- features ----
    Xtr, y_sro_tr, y_ph_tr = seq_dataset.flat_features(train_idx)
    Xte, y_sro_te, y_ph_te = seq_dataset.flat_features(test_idx)

    ns = seq_dataset.tokenizer.n_species
    # Composition-only features = last ns columns of X
    Xcomp_tr = Xtr[:, -ns:]
    Xcomp_te = Xte[:, -ns:]

    results = {'sro': {}, 'phase': {}}

    # ---- SRO regression ----
    sro_models = [
        ('Ridge (comp)',     _reg_pipeline(Ridge(alpha=1.0)),
         Xcomp_tr, Xcomp_te),
        ('MLP (comp)',       _reg_pipeline(MLPRegressor(
             hidden_layer_sizes=(64,64), max_iter=400,
             random_state=42, early_stopping=True)),
         Xcomp_tr, Xcomp_te),
        ('MLP (local-env)',  _reg_pipeline(MLPRegressor(
             hidden_layer_sizes=(128,128,64), max_iter=600,
             random_state=42, early_stopping=True,
             learning_rate_init=5e-4)),
         Xtr, Xte),
        ('RandomForest',     Pipeline([('m', RandomForestRegressor(
             n_estimators=100, n_jobs=-1, random_state=42,
             min_samples_leaf=2))]),
         Xtr, Xte),
    ]
    for name, model, Xt, Xv in sro_models:
        if verbose:
            print(f"  [SRO] {name} ...", end=' ', flush=True)
        try:
            model.fit(Xt, y_sro_tr)
            m = _eval_reg(model, Xv, y_sro_te)
            results['sro'][name] = m
            if verbose:
                print(f"MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
        except Exception as e:
            if verbose: print(f"FAIL: {e}")
            results['sro'][name] = {}

    # ---- Phase classification ----
    ph_models = [
        ('Ridge (comp)',    _cls_pipeline(LogisticRegression(
             max_iter=500, random_state=42)), Xcomp_tr, Xcomp_te),
        ('MLP (comp)',      _cls_pipeline(MLPClassifier(
             hidden_layer_sizes=(64,32), max_iter=400,
             random_state=42, early_stopping=True)), Xcomp_tr, Xcomp_te),
        ('MLP (local-env)', _cls_pipeline(MLPClassifier(
             hidden_layer_sizes=(64,64,32), max_iter=400,
             random_state=42, early_stopping=True)), Xtr, Xte),
        ('RandomForest',    _cls_pipeline(RandomForestClassifier(
             n_estimators=100, n_jobs=-1, random_state=42)), Xtr, Xte),
    ]
    for name, model, Xt, Xv in ph_models:
        if verbose:
            print(f"  [Phase] {name} ...", end=' ', flush=True)
        try:
            model.fit(Xt, y_ph_tr)
            m = _eval_cls(model, Xv, y_ph_te)
            results['phase'][name] = m
            if verbose:
                print(f"Acc={m['acc']:.4f}  F1={m['f1']:.4f}")
        except Exception as e:
            if verbose: print(f"FAIL: {e}")
            results['phase'][name] = {}

    return results


def frozen_transformer_baseline(encoder, seq_dataset,
                                  train_idx, test_idx,
                                  verbose=True) -> dict:
    """
    Extract mean-pooled encoder embeddings, then fit Ridge + LogReg.
    Tests whether the (pre-trained) representation is linearly separable.
    """
    def extract(indices):
        feats = []
        for idx in indices:
            s = seq_dataset[idx]
            enc, _ = encoder.forward(s['token_ids'], s['features'])
            feats.append(enc.mean(0))
        return np.array(feats)

    if verbose:
        print("  [FrozenEncoder] extracting features ...", end=' ', flush=True)

    Xtr = extract(train_idx);  Xte = extract(test_idx)
    _, y_sro_tr, y_ph_tr = seq_dataset.flat_features(train_idx)
    _, y_sro_te, y_ph_te = seq_dataset.flat_features(test_idx)

    if verbose: print("done")

    results = {}

    # SRO
    m = _reg_pipeline(Ridge(alpha=1.0))
    m.fit(Xtr, y_sro_tr)
    r = _eval_reg(m, Xte, y_sro_te)
    results['sro_frozen+Ridge'] = r
    if verbose:
        print(f"  [FrozenEncoder+Ridge SRO] MAE={r['mae']:.4f}  R²={r['r2']:.4f}")

    # Phase
    c = _cls_pipeline(LogisticRegression(max_iter=500, random_state=42))
    c.fit(Xtr, y_ph_tr)
    rc = _eval_cls(c, Xte, y_ph_te)
    results['phase_frozen+LR'] = rc
    if verbose:
        print(f"  [FrozenEncoder+LR Phase] Acc={rc['acc']:.4f}  F1={rc['f1']:.4f}")

    return results
