import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AUC_FAIL_THRESHOLD = 0.75
CRAMERS_V_FAIL_THRESHOLD = 0.5


def univariate_auc(feature: pd.Series, label: pd.Series) -> float:
    score = feature.astype("float64")
    auc = roc_auc_score(label, score)
    # Direction-agnostic: a field that predicts "not fraud" strongly is just as leaky
    # as one that predicts "fraud" strongly.
    return max(auc, 1 - auc)


def _chi2_statistic(contingency: pd.DataFrame) -> float:
    observed = contingency.to_numpy(dtype="float64")
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums @ col_sums / total
    return float(((observed - expected) ** 2 / expected).sum())


def cramers_v(feature: pd.Series, label: pd.Series) -> float:
    contingency = pd.crosstab(feature, label)
    chi2 = _chi2_statistic(contingency)
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    denom = min(r - 1, k - 1)
    if denom == 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))
