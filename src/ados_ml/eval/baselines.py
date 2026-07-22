"""Sklearn baselines on concatenated task features."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ados_ml.eval.metrics import binary_metrics, ordinal_metrics, regression_metrics


def _flat(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def run_sklearn_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: dict[str, np.ndarray],
    y_test: dict[str, np.ndarray],
) -> dict[str, Any]:
    Xt = _flat(X_train)
    Xe = _flat(X_test)
    results: dict[str, Any] = {}

    # SA / RRB ridge
    for name in ("SA", "RRB"):
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        pipe.fit(Xt, y_train[name])
        pred = pipe.predict(Xe)
        results[name] = regression_metrics(y_test[name], pred)

    # B1 logistic
    pipe_b1 = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    pipe_b1.fit(Xt, y_train["B1"].astype(int))
    prob = pipe_b1.predict_proba(Xe)[:, 1]
    pred = (prob >= 0.5).astype(float)
    results["B1"] = binary_metrics(y_test["B1"], prob, pred)

    # C2 / B12 as multinomial logistic (levels as classes)
    for name, n_levels in (("C2", 4), ("B12", 3)):
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        yt = y_train[name].astype(int)
        pipe.fit(Xt, yt)
        pred = pipe.predict(Xe).astype(float)
        # expected value from proba if available
        if hasattr(pipe[-1], "predict_proba"):
            # pipeline predict_proba
            proba = pipe.predict_proba(Xe)
            classes = pipe[-1].classes_
            exp = (proba * classes.reshape(1, -1)).sum(axis=1)
        else:
            exp = pred
        results[name] = ordinal_metrics(y_test[name], pred, exp)

    return results
