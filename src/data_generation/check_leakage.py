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


from pathlib import Path

FIELD_METADATA: list[dict] = [
    {
        "field_name": "hour_of_day", "data_type": "int16", "unit": "hour (0-23)",
        "valid_range": "[0, 23]", "generation_type": "derived",
        "logic_or_formula": "(step - 1) % 24",
        "business_assumption": "step in PaySim represents elapsed hours since simulation start.",
        "metric_type": "auc",
    },
    {
        "field_name": "is_night_transaction", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "derived",
        "logic_or_formula": "hour_of_day in [0, 5]",
        "business_assumption": "Night defined as 00:00-05:59.",
        "metric_type": "auc",
    },
    {
        "field_name": "customer_account_age_days", "data_type": "int32", "unit": "days",
        "valid_range": "[1, 3650]", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "lognormal(median=400 base / 150 fraud, sigma=0.6), clipped",
        "business_assumption": "Compromised/mule accounts assumed more often recently created (business assumption, not empirically derived).",
        "metric_type": "auc",
    },
    {
        "field_name": "device_id", "data_type": "string", "unit": "UUID",
        "valid_range": "50,000-value pool", "generation_type": "independent-random",
        "logic_or_formula": "uniform sample from a fixed 50,000-UUID Faker-generated pool",
        "business_assumption": "Device identity alone carries no fraud signal; signal lives in new_device_flag.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "browser", "data_type": "category", "unit": "categorical",
        "valid_range": "{Chrome, Safari, Edge, Firefox, Other}", "generation_type": "independent-random",
        "logic_or_formula": "weighted categorical: Chrome 55%, Safari 20%, Edge 12%, Firefox 8%, Other 5%",
        "business_assumption": "No behavioral link to fraud assumed; kept neutral to avoid manufactured signal.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "device_type", "data_type": "category", "unit": "categorical",
        "valid_range": "{mobile, desktop, tablet}", "generation_type": "independent-random",
        "logic_or_formula": "weighted categorical: mobile 65%, desktop 30%, tablet 5%",
        "business_assumption": "No behavioral link to fraud assumed.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "new_device_flag", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "Bernoulli(p=0.04 base / 0.12 fraud)",
        "business_assumption": "Account-takeover fraud plausibly more often from an unrecognized device; odds-ratio capped at 3x (business assumption).",
        "metric_type": "auc",
    },
    {
        "field_name": "billing_country", "data_type": "category", "unit": "ISO country code",
        "valid_range": "20 fixed countries", "generation_type": "independent-random",
        "logic_or_formula": "weighted categorical over 20 countries (simulated customer base geography)",
        "business_assumption": "No direct fraud link; signal lives in ip_country mismatch.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "ip_country", "data_type": "category", "unit": "ISO country code",
        "valid_range": "20 fixed countries", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "= billing_country with p=0.93 base / 0.80 fraud, else a different country",
        "business_assumption": "Legit traffic mostly matches home country; fraud has higher (but not certain) mismatch odds since VPNs let fraud spoof location too.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "ip_billing_distance_km", "data_type": "float64", "unit": "km",
        "valid_range": "[0, ~20000]", "generation_type": "derived",
        "logic_or_formula": "haversine(centroid[ip_country], centroid[billing_country])",
        "business_assumption": "Derived directly from ip_country/billing_country for internal consistency.",
        "metric_type": "auc",
    },
    {
        "field_name": "shipping_billing_mismatch", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "Bernoulli(p=0.05 base / 0.15 fraud)",
        "business_assumption": "Reinterpreted as 'transaction address differs from registered address' given PaySim's account-takeover fraud pattern (not card-present checkout).",
        "metric_type": "auc",
    },
    {
        "field_name": "failed_payment_attempts_24h", "data_type": "int16", "unit": "count",
        "valid_range": "[0, ~10]", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "Poisson(lambda=0.15 base / 0.6 fraud)",
        "business_assumption": "Attackers often attempt multiple times before succeeding; odds-ratio capped at 4x (business assumption).",
        "metric_type": "auc",
    },
]


def check_all_fields(df: pd.DataFrame, label_col: str = "isFraud") -> pd.DataFrame:
    label = df[label_col]
    rows = []
    for meta in FIELD_METADATA:
        name = meta["field_name"]
        if meta["metric_type"] == "auc":
            value = univariate_auc(df[name], label)
            status = "FAIL" if value >= AUC_FAIL_THRESHOLD else "PASS"
        else:
            value = cramers_v(df[name], label)
            status = "FAIL" if value >= CRAMERS_V_FAIL_THRESHOLD else "PASS"
        rows.append({
            "field_name": name,
            "metric_type": meta["metric_type"],
            "metric_value": round(value, 4),
            "status": status,
        })
    return pd.DataFrame(rows)


def build_data_dictionary_markdown(leakage_report: pd.DataFrame) -> str:
    metric_by_field = leakage_report.set_index("field_name")["metric_value"].to_dict()
    lines = [
        "# Data Dictionary — Synthetic Contextual Fields\n\n",
        "| field_name | data_type | unit | valid_range | generation_type | logic_or_formula | "
        "business_assumption | measured_univariate_association_with_isFraud |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for meta in FIELD_METADATA:
        metric = metric_by_field.get(meta["field_name"], "n/a")
        lines.append(
            f"| `{meta['field_name']}` | {meta['data_type']} | {meta['unit']} | {meta['valid_range']} | "
            f"{meta['generation_type']} | {meta['logic_or_formula']} | {meta['business_assumption']} | {metric} |\n"
        )
    return "".join(lines)


def main():
    df = pd.read_parquet("data/processed/transactions_synthetic.parquet")
    report = check_all_fields(df)
    print(report.to_string(index=False))
    failures = report[report["status"] == "FAIL"]
    if len(failures) > 0:
        raise SystemExit(f"Leakage check FAILED for fields: {list(failures['field_name'])}")
    markdown = build_data_dictionary_markdown(report)
    Path("docs/DATA_DICTIONARY.md").write_text(markdown, encoding="utf-8")
    print("Wrote docs/DATA_DICTIONARY.md")


if __name__ == "__main__":
    main()
