from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def fit_logistic_model(
    X: NDArray[Any],
    y: NDArray[Any],
    sample_weight: NDArray[Any],
    regularization: float,
    use_scaling: bool = True,
    cv_folds: int = 5,
) -> tuple[LogisticRegression, StandardScaler | None, dict[str, Any]]:
    """Fit logistic regression model with optional scaling and cross-validation.
    
    Args:
        X: Feature matrix (typically log-transformed human times)
        y: Success rates (0-1, can be fractional)
        sample_weight: Sample weights
        regularization: L2 regularization strength (inverse of C)
        use_scaling: Whether to standardize features
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (fitted model, scaler, metrics dict)
    """
   
    original_weight_sum = np.sum(sample_weight)
    original_average = np.average(y, weights=sample_weight)

    # Split fractional values into weighted 0s and 1s
    fractional_mask = (y > 0) & (y < 1)
    if np.any(fractional_mask):
        X_frac = X[fractional_mask]
        y_frac = y[fractional_mask]
        w_frac = sample_weight[fractional_mask]

        X_split = np.vstack([X_frac, X_frac])
        y_split = np.zeros(2 * len(y_frac))
        y_split[len(y_frac):] = 1
        w_split = np.concatenate([(1 - y_frac) * w_frac, y_frac * w_frac])

        X = np.vstack([X[~fractional_mask], X_split])
        y = np.concatenate([y[~fractional_mask], y_split])
        sample_weight = np.concatenate([sample_weight[~fractional_mask], w_split])
        
        assert np.allclose(np.sum(sample_weight), original_weight_sum)
        assert np.allclose(np.average(y, weights=sample_weight), original_average)


    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)


    model = LogisticRegression(C=1 / regularization, max_iter=1000)
    model.fit(X, y, sample_weight=sample_weight)


    y_pred_proba = model.predict_proba(X)[:, 1]
    train_loss = log_loss(y, y_pred_proba, sample_weight=sample_weight)
    
    # Cross-validation
    cv_scores = None
    try:
        cv_scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring='neg_log_loss'
        )
    except Exception:
        pass

    metrics = {
        'training_log_loss': train_loss,
        'cv_mean_log_loss': -cv_scores.mean() if cv_scores is not None else None,
        'cv_std_log_loss': cv_scores.std() if cv_scores is not None else None,
    }

    return model, scaler, metrics


def get_time_horizon(
    model: LogisticRegression,
    scaler: StandardScaler | None,
    success_rate: float
) -> float:
    
    log_odds = np.log(success_rate / (1 - success_rate))
    feature_value = (log_odds - model.intercept_[0]) / model.coef_[0][0]
    
    if scaler is not None:
        feature_value = scaler.inverse_transform([[feature_value]])[0, 0]
        
    return feature_value


def fit_math_olympiad_model(
    df: pd.DataFrame,
    time_col: str,
    success_col: str,
    weight_col: str,
    log_transform: bool = True,
    regularization: float = 0.01,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Fit logistic regression model to math olympiad data.
    
    Args:
        df: DataFrame with evaluation results
        time_col: Column name for human completion time
        success_col: Column name for success indicator (0/1)
        weight_col: Column name for sample weights
        log_transform: Whether to log-transform time values
        regularization: L2 regularization strength
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with model summary and time horizons
    """

    times = df[time_col].values
    success = df[success_col].values
    weights = df[weight_col].values
    
    valid_mask = times > 0
    times = times[valid_mask]
    success = success[valid_mask]
    weights = weights[valid_mask]
    
    if log_transform:
        X = np.log(times).reshape(-1, 1)
        feature_names = ['log_human_minutes']
    else:
        X = times.reshape(-1, 1)
        feature_names = ['human_minutes']
    
    model, scaler, metrics = fit_logistic_model(
        X, success, weights, regularization, 
        use_scaling=True, cv_folds=cv_folds
    )
    
    horizons = {}
    for rate in [0.25, 0.5, 0.75, 0.9]:
        try:
            horizon = get_time_horizon(model, scaler, rate)
            if log_transform:
                horizon = np.exp(horizon)
            horizons[f'time_horizon_{int(rate*100)}pct'] = float(horizon)
        except Exception:
            horizons[f'time_horizon_{int(rate*100)}pct'] = None
    
    summary = {
        'regularization': regularization,
        'use_scaling': True,
        'cv_folds': cv_folds,
        'training_log_loss': metrics['training_log_loss'],
        'cv_mean_log_loss': metrics['cv_mean_log_loss'],
        'cv_std_log_loss': metrics['cv_std_log_loss'],
        'coefficients': model.coef_[0].tolist(),
        'intercept': float(model.intercept_[0]),
        'feature_names': feature_names,
        'n_features': X.shape[1],
        **horizons
    }
    
    return summary
