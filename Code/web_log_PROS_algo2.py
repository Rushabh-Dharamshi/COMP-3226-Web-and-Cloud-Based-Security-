import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_FILE = "processed_web_log_features.csv"
DIST_DIR = "clean_distributions"
OUTPUT_FILE = "scored_traffic.csv"
ROC_PLOT_FILE = "roc_curve_comparison.png"

# Load Data
print(f"Loading traffic data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
df.fillna("Unknown", inplace=True)

# ==============================================================================
# 1. LOAD CLEAN DISTRIBUTIONS (The "Brain")
# ==============================================================================
dist_cache = {}

print("Loading clean distributions...")
csv_files = glob.glob(os.path.join(DIST_DIR, "*.csv"))

for f in csv_files:
    filename = os.path.basename(f)
    parts = filename.replace("clean_", "").replace(".csv", "").split("_by_")
    if len(parts) != 2: continue
    
    target, condition = parts
    prob_df = pd.read_csv(f, index_col=0)
    dist_cache[target] = prob_df.to_dict()
    print(f"  Loaded probabilities for '{target}' (conditioned on '{condition}')")

# Helper function to safely look up probabilities
def get_clean_prob(target_name, condition_val, feature_val):
    try:
        if condition_val not in dist_cache[target_name]:
            condition_val = "Unknown" 
            # Fallback if even Unknown is missing
            if condition_val not in dist_cache[target_name]:
                return 1e-9
            
        probs = dist_cache[target_name].get(condition_val, {})
        p = probs.get(str(feature_val), 0.0)
        return max(p, 1e-9)
    except Exception:
        return 1e-9

# ==============================================================================
# 2. CALCULATE PROS SCORES (Algorithm 2)
# ==============================================================================
print("Calculating Clean Probabilities (PROS Algorithm)...")

def get_clean_vector(df, target, condition_col):
    def lookup(row):
        cond = row[condition_col]
        val = row[target]
        return get_clean_prob(target, cond, val)
    return df.apply(lookup, axis=1)

# 3. Lookup Clean Probabilities for Each Feature
print("  > Looking up P(path | country)...")
df['p_clean_path'] = get_clean_vector(df, 'path_feature', 'country')

print("  > Looking up P(status | country)...")
df['p_clean_status'] = get_clean_vector(df, 'status', 'country')

print("  > Looking up P(browser | family)...")
df['p_clean_browser'] = get_clean_vector(df, 'browser', 'family')

print("  > Looking up P(family | country)...")
df['p_clean_family'] = get_clean_vector(df, 'family', 'country')


# Calculate Joint Scores
print("Computing Joint Probability P(Clean)...")
mask_modern = df['family'].isin(['Chrome', 'Firefox', 'Mobile Safari', 'Chrome Mobile'])

# Group A Score
df.loc[mask_modern, 'P_clean_joint'] = (
    df.loc[mask_modern, 'p_clean_browser'] * df.loc[mask_modern, 'p_clean_path'] * df.loc[mask_modern, 'p_clean_status']
)
# Group B Score
df.loc[~mask_modern, 'P_clean_joint'] = (
    df.loc[~mask_modern, 'p_clean_family'] * df.loc[~mask_modern, 'p_clean_path'] * df.loc[~mask_modern, 'p_clean_status']
)
# Define the tuple for Observed Count
cols_A = ['browser', 'path_feature', 'status']

# Calculate Observed Frequencies
print("Computing Observed Frequencies...")
cols_A = ['browser', 'path_feature', 'status']
counts_A = df[mask_modern].groupby(cols_A).size().reset_index(name='count_A')
counts_A['P_obs_joint'] = counts_A['count_A'] / counts_A['count_A'].sum()
df = df.merge(counts_A, on=cols_A, how='left')

cols_B = ['family', 'path_feature', 'status']
counts_B = df[~mask_modern].groupby(cols_B).size().reset_index(name='count_B')
counts_B['P_obs_joint_B'] = counts_B['count_B'] / counts_B['count_B'].sum()

df_B_merged = df[~mask_modern].merge(counts_B, on=cols_B, how='left')
df.loc[~mask_modern, 'P_obs_joint'] = df_B_merged['P_obs_joint_B'].values

# Final Odds Calculation
print("Calculating Final Odds...")
alpha = 0.5
df['bot_odds'] = df['P_obs_joint'] / (alpha * df['P_clean_joint']) - 1
df['bot_odds'].fillna(-1, inplace=True) # Handle artifacts

# Save PROS Results
df.sort_values(by='bot_odds', ascending=False, inplace=True)

# Save
out_cols = [
    'ip', 'timestamp', 'request', 'status', 'user_agent', # Raw info
    'browser', 'family', 'path_feature', 'osFamily', 'state',                 # Features
    'label',                                              # Ground Truth (from prev step)
    'bot_odds'                                            # The Score
]

# Ensure we have all columns (some might be missing if raw csv not fully loaded)
# Assuming 'timestamp' and 'request' are in the input CSV. 
# If not, use the columns we have.
available_cols = [c for c in out_cols if c in df.columns]
df[available_cols].to_csv(OUTPUT_FILE, index=False)

print("-" * 60)
print(f"DONE. Scored traffic saved to {OUTPUT_FILE}")
print("Top 5 Most Suspicious Requests:")
print(df[available_cols].head(5))
print("-" * 60)

# ==============================================================================
# 3. TRAIN ISOLATION FOREST (Baseline Comparison)
# ==============================================================================
print("\nTraining Isolation Forest for comparison...")
iso_features = ['family', 'path_feature', 'status']

# One-Hot Encoding
try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

X_iso = encoder.fit_transform(df[iso_features].astype(str))

# Train Model
clf = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
clf.fit(X_iso)

# Get Scores (Negate so higher = more anomalous)
iso_scores = -clf.decision_function(X_iso)

# ==============================================================================
# 4. PLOT ROC CURVES
# ==============================================================================
print("Plotting ROC Curves...")

# Calculate ROC for PROS
fpr_pros, tpr_pros, _ = roc_curve(df['label'], df['bot_odds'])
roc_auc_pros = auc(fpr_pros, tpr_pros)

# Calculate ROC for Isolation Forest
fpr_iso, tpr_iso, _ = roc_curve(df['label'], iso_scores)
roc_auc_iso = auc(fpr_iso, tpr_iso)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr_pros, tpr_pros, color='darkorange', lw=2, label=f'PROS (AUC = {roc_auc_pros:.3f})')
plt.plot(fpr_iso, tpr_iso, color='navy', lw=2, linestyle='--', label=f'Isolation Forest (AUC = {roc_auc_iso:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: PROS vs. Isolation Forest')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.savefig(ROC_PLOT_FILE)
print(f"ROC Curve saved to {ROC_PLOT_FILE}")

print("-" * 60)
print(f"PROS AUC: {roc_auc_pros:.4f}")
print(f"Isolation Forest AUC: {roc_auc_iso:.4f}")
print("-" * 60)