# make_labels.py
# Merge objective features (features.csv) with questionnaire scores (responses.csv)
# and create dataset.csv with a risk label derived from OSDI.

import pandas as pd

# ---- thresholds you can adjust if your rubric differs ----
def osdi_to_label(osdi):
    # Example mapping (common in literature):
    # 0–12: Low, 13–22: Medium, 23–100: High
    if osdi >= 23:
        return "High"
    if osdi >= 13:
        return "Medium"
    return "Low"

# Read inputs (must be in the same folder)
features = pd.read_csv("features.csv")
responses = pd.read_csv("responses.csv")

# Ensure expected columns exist
required_cols = {"participant_id", "osdi_score"}
missing = required_cols - set(responses.columns)
if missing:
    raise ValueError(f"responses.csv is missing columns: {missing}. "
                     f"Expected exactly: {required_cols}")

# Map score -> label
responses["risk_label"] = responses["osdi_score"].apply(osdi_to_label)

# Merge on participant_id
dataset = features.merge(
    responses[["participant_id", "osdi_score", "risk_label"]],
    on="participant_id",
    how="inner"
)

# Save final dataset
dataset.to_csv("dataset.csv", index=False)
print("[OK] Saved dataset.csv with", len(dataset), "rows")
