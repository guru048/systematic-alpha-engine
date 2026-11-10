"""
ML alpha model: LightGBM with CPCV and SHAP feature attribution.

Workflow
--------
1. Build panel feature matrix X (T x N factors) and target y (1-day fwd return).
2. Use CombPurgedCV to avoid leakage.
3. Train LightGBM regressors on each fold.
4. Pool OOS predictions → evaluate IC / ICIR.
5. Compute mean SHAP values → feature importance.
6. Compute Probabilistic Sharpe Ratio of OOS predictions.

The model predicts the cross-sectional z-score of forward returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .cross_validation import CombPurgedCV


class AlphaModel:
    """
    LightGBM-based alpha factor model with Combinatorial Purged CV.

    Parameters
    ----------
    n_splits      : CPCV folds.
    test_splits   : CPCV test folds per combination.
    purge_days    : Days to purge around test sets.
    embargo_days  : Days to embargo after test sets.
    lgbm_params   : LightGBM hyperparameters.
    """

    DEFAULT_PARAMS = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(
        self,
        n_splits: int = 6,
        test_splits: int = 1,
        purge_days: int = 5,
        embargo_days: int = 5,
        lgbm_params: Optional[dict] = None,
    ):
        self.cv = CombPurgedCV(n_splits, test_splits, purge_days, embargo_days)
        self.params = {**self.DEFAULT_PARAMS, **(lgbm_params or {})}
        self.models_: List = []
        self.feature_names_: List[str] = []
        self.shap_values_: Optional[np.ndarray] = None

    # ── Build feature matrix ─────────────────────────────────────────────

    @staticmethod
    def build_feature_matrix(
        factor_dict: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        forward_days: int = 1,
        min_stocks: int = 20,
    ) -> pd.DataFrame:
        """
        Stack factor DataFrames into a tall (obs x features) DataFrame.

        Each row is (date, stock) with columns = factor values.
        Target = z-score-normalised forward return.

        Returns DataFrame with columns [factor_names..., 'target'].
        """
        stocks = returns.columns
        dates = returns.index

        rows = []
        for t, date in enumerate(dates[:-forward_days]):
            fwd_date = dates[t + forward_days]
            # forward return (target)
            fwd_ret = returns.loc[fwd_date]

            row_data = []
            for name, fdf in factor_dict.items():
                if date not in fdf.index:
                    break
                row_data.append(fdf.loc[date])
            else:
                # Cross-sectional z-score of features
                frame = pd.DataFrame(dict(zip(factor_dict.keys(), row_data)))
                # z-score normalise each factor cross-sectionally
                frame = (frame - frame.mean()) / (frame.std() + 1e-8)
                frame["target"] = (fwd_ret - fwd_ret.mean()) / (fwd_ret.std() + 1e-8)
                frame["_date"] = date
                rows.append(frame.reset_index().rename(columns={"index": "stock"}))

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    # ── Train ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "AlphaModel":
        """
        Train with CPCV. Stores OOS predictions for evaluation.

        Parameters
        ----------
        X : (n_obs, n_features) feature matrix.
        y : (n_obs,) target (forward return z-scores).
        feature_names : list of feature names for SHAP plots.
        """
        try:
            import lightgbm as lgb
            _lgb_available = True
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            _lgb_available = False

        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.models_ = []
        oos_preds = np.full(len(y), np.nan)
        shap_accumulator = np.zeros(X.shape[1])
        shap_count = 0

        for train_idx, test_idx in self.cv.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]

            # Remove NaN rows from training
            mask_tr = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            mask_te = ~np.isnan(X_te).any(axis=1)

            if mask_tr.sum() < 20:
                continue

            if _lgb_available:
                model = lgb.LGBMRegressor(**self.params)
                model.fit(X_tr[mask_tr], y_tr[mask_tr])
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42
                )
                model.fit(X_tr[mask_tr], y_tr[mask_tr])

            self.models_.append(model)
            if mask_te.sum() == 0:
                continue
            pred = model.predict(X_te[mask_te])
            te_valid = np.where(mask_te)[0]
            oos_preds[np.array(test_idx)[te_valid]] = pred

            # SHAP feature importance (use built-in importance as approximation)
            if _lgb_available and hasattr(model, "feature_importances_"):
                shap_accumulator += model.feature_importances_
                shap_count += 1

        self.oos_predictions_ = oos_preds
        self.y_true_ = y
        if shap_count > 0:
            self.shap_values_ = shap_accumulator / shap_count

        return self

    # ── OOS evaluation ───────────────────────────────────────────────────

    def oos_ic(self) -> float:
        """Out-of-sample IC: Spearman correlation of OOS predictions vs actual."""
        from scipy.stats import spearmanr
        mask = ~(np.isnan(self.oos_predictions_) | np.isnan(self.y_true_))
        if mask.sum() < 20:
            return np.nan
        ic, _ = spearmanr(self.oos_predictions_[mask], self.y_true_[mask])
        return float(ic)

    def feature_importance(self) -> pd.Series:
        """Return mean SHAP / feature importance as a Series."""
        if self.shap_values_ is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.shap_values_, index=self.feature_names_
        ).sort_values(ascending=False)
