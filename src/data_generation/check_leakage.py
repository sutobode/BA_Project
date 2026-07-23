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
    """Bias-corrected Cramer's V (Bergsma, 2013).

    The naive/uncorrected formula sqrt((chi2/n) / min(r-1, k-1)) is biased
    upward for sparse contingency tables (e.g. high-cardinality categorical
    fields on modest sample sizes), which can produce spurious high scores
    even when the feature has zero true association with the label. The
    bias correction subtracts the expected small-sample noise from phi2 and
    the row/column counts before taking the ratio.
    """
    contingency = pd.crosstab(feature, label)
    chi2 = _chi2_statistic(contingency)
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_corrected - 1, r_corrected - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denom))


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
        "valid_range": "[1, 3650]", "generation_type": "conditional-on-risk-proxy",
        "logic_or_formula": "lognormal(median=400 at risk_score=0 / 275 at risk_score=1, sigma=0.6), clipped; risk_score = compute_risk_proxy(type, amount, hour_of_day)",
        "business_assumption": "Compromised/mule accounts assumed more often recently created. Driven by a label-free observable risk proxy (transaction channel, amount percentile within channel, night-time), never by isFraud, so the value is computable for a brand-new transaction at scoring time.",
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
        "valid_range": "{True, False}", "generation_type": "conditional-on-risk-proxy",
        "logic_or_formula": "Bernoulli(p = 0.04 + risk_score * (0.12 - 0.04)); risk_score = compute_risk_proxy(type, amount, hour_of_day)",
        "business_assumption": "Account-takeover fraud plausibly more often from an unrecognized device; odds-ratio capped at 3x (business assumption). Driven by the label-free risk proxy, not isFraud.",
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
        "valid_range": "20 fixed countries", "generation_type": "conditional-on-risk-proxy",
        "logic_or_formula": "= billing_country with p = 0.93 + risk_score * (0.80 - 0.93), else a different country; risk_score = compute_risk_proxy(type, amount, hour_of_day)",
        "business_assumption": "Legit traffic mostly matches home country; higher-risk transactions have higher (but not certain) mismatch odds since VPNs let fraud spoof location too. Driven by the label-free risk proxy, not isFraud.",
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
        "field_name": "ip_billing_country_mismatch", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "derived",
        "logic_or_formula": "ip_country != billing_country",
        "business_assumption": "Boolean convenience view of the same ip_country/billing_country mismatch already captured numerically by ip_billing_distance_km; provided for teams that need a flag rather than a distance.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "shipping_billing_mismatch", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "conditional-on-risk-proxy",
        "logic_or_formula": "Bernoulli(p = 0.05 + risk_score * (0.15 - 0.05)); risk_score = compute_risk_proxy(type, amount, hour_of_day)",
        "business_assumption": "Reinterpreted as 'transaction address differs from registered address' given PaySim's account-takeover fraud pattern (not card-present checkout). Driven by the label-free risk proxy, not isFraud.",
        "metric_type": "auc",
    },
    {
        "field_name": "failed_payment_attempts_24h", "data_type": "int16", "unit": "count",
        "valid_range": "[0, ~10]", "generation_type": "conditional-on-risk-proxy",
        "logic_or_formula": "Poisson(lambda = 0.15 + risk_score * (0.6 - 0.15)); risk_score = compute_risk_proxy(type, amount, hour_of_day)",
        "business_assumption": "Attackers often attempt multiple times before succeeding; odds-ratio capped at 4x (business assumption). Driven by the label-free risk proxy, not isFraud.",
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
    # Build lookup dicts for both metric_value and metric_type
    metric_by_field = leakage_report.set_index("field_name")["metric_value"].to_dict()
    metric_type_by_field = leakage_report.set_index("field_name")["metric_type"].to_dict()
    lines = [
        "# Data Dictionary — Synthetic Contextual Fields\n\n",
        "| field_name | data_type | unit | valid_range | generation_type | logic_or_formula | "
        "business_assumption | measured_univariate_association_with_isFraud |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for meta in FIELD_METADATA:
        metric_value = metric_by_field.get(meta["field_name"], "n/a")
        metric_type = metric_type_by_field.get(meta["field_name"], "unknown")

        # Format metric with type label
        if metric_value == "n/a":
            metric_str = "n/a"
        elif metric_type == "auc":
            metric_str = f"{metric_value} (AUC)"
        elif metric_type == "cramers_v":
            metric_str = f"{metric_value} (Cramér's V)"
        else:
            metric_str = str(metric_value)

        lines.append(
            f"| `{meta['field_name']}` | {meta['data_type']} | {meta['unit']} | {meta['valid_range']} | "
            f"{meta['generation_type']} | {meta['logic_or_formula']} | {meta['business_assumption']} | {metric_str} |\n"
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
