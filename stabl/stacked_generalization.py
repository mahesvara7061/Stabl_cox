# __LIBRARIES__

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, r2_score
# >>> SURVIVAL PATCH: imports
from sksurv.metrics import concordance_index_censored
import warnings



# __FUNCTIONS__

def stacked_multi_omic(df_predictions, y, task_type, n_iter=10000):
    """
    Functions to compute the stacked generalization using the prediction of 
    models trained on individual omics.

    The function automatically handles missing values.

    Parameters
    ----------
    df_predictions: pd.DataFrame
        The DataFrame containing all the predictions for each omic.

    y: pd.Series
        pandas Series containing all the outcomes.

    task_type: string
        "binary" or "regression"

    n_iter: int
        Number of iterations to perform; each iteration corresponds to a random search 
        of weights to test.

    Returns
    -------
    df_predictions: pd.DataFrame
        pandas DataFrame containing the predictions for each omic as well as the 
        weighted stacked generalization predictions

    df_weights: pd.DataFrame
        pandas DataFrame containing the final weights associated to each omic.
    """

    df_predictions = df_predictions.drop(columns=y.name, errors="ignore")

    best_score = -100
    # best_weights = []
    # best_probs = []
    best_weights = None
    best_probs = None

    # if task_type == "survival":
    #     # y bắt buộc có cột 'time' và 'event'
    #     if not isinstance(y, (pd.Series, pd.DataFrame)) or not set(["time", "event"]).issubset(set(y.columns)):
    #         raise ValueError("For survival, y must be a DataFrame with columns ['time','event'].")
    if task_type == "survival":
        if isinstance(y, pd.Series):
            raise ValueError("For survival, y must be a DataFrame, not Series")
        if not isinstance(y, pd.DataFrame):
            raise ValueError("For survival, y must be a DataFrame")
        if not set(["time", "event"]).issubset(set(y.columns)):
            raise ValueError(f"Missing columns. Need ['time','event'], got: {list(y.columns)}")
        # Đồng bộ index để tránh lệch
        # y_surv = y.loc[df_predictions.index]
        # ev_all = y_surv["event"].astype(bool).to_numpy()
        # tt_all = y_surv["time"].to_numpy()
        # ✅ CORRECTED CODE
# Ensure alignment before converting to numpy
        y_surv = y.loc[df_predictions.index].copy()
        mask_valid = (~df_predictions.isna().all(axis=1)).to_numpy()
        ev_all = y_surv.loc[mask_valid, "event"].astype(bool).to_numpy()
        tt_all = y_surv.loc[mask_valid, "time"].to_numpy()
        weighted_probs_valid = weighted_probs[mask_valid].to_numpy()

    for i in range(n_iter):
        weights = np.random.uniform(0, 10, df_predictions.shape[1])
        weighted_probs = ((df_predictions * weights).sum(1)) / ((~df_predictions.isna() * weights).sum(1))
        if task_type == "binary":
            try:
                score = roc_auc_score(y, weighted_probs)
            except:
                continue
        elif task_type == "regression":
            try:
                score = r2_score(y, weighted_probs)
            except:
                continue

        elif task_type == "survival":
            # Loại NaN trước khi tính C-index
            mask = ~weighted_probs.isna().to_numpy()
            if mask.sum() < 2:
                continue
            try:
                score = concordance_index_censored(ev_all[mask], tt_all[mask], weighted_probs.to_numpy()[mask])[0]
            except Exception:
                continue


        else:
            raise ValueError("task_type not recognized")

        if score > best_score:
            best_probs = weighted_probs
            best_score = score
            best_weights = weights

    if best_probs is None or best_weights is None:
        warnings.warn(f"No valid weights found. Using uniform weights.", UserWarning)
        weights = np.ones(df_predictions.shape[1])
        denom = ((~df_predictions.isna()) * weights).sum(axis=1)
        best_probs = ((df_predictions * weights).sum(axis=1)) / denom.replace(0, np.nan)
        best_weights = weights

    df_weights = pd.DataFrame(
        data={"Associated weight": best_weights},
        index=df_predictions.columns
    )
    df_predictions["Stacked Gen. Predictions"] = best_probs
    # Gắn lại outcome cho tiện theo dõi/lưu CSV
    if isinstance(y, pd.Series):
        df_predictions[y.name] = y
    else:
        # survival: thêm cả hai cột time/event (nếu trùng tên cột sẽ tự override)
        df_predictions = df_predictions.join(y.loc[df_predictions.index], how="left")


    return df_predictions, df_weights
