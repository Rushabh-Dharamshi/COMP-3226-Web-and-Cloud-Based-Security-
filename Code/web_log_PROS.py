import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = "processed_web_log_features.csv"
OUTPUT_DIR = "clean_distributions"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Fill NaNs to ensure grouping works
df.fillna("Unknown", inplace=True)

# OPTIONAL: Reduce cardinality for noisy features to make matrices manageable
def reduce_cardinality(series, top_n=50):
    counts = series.value_counts()
    top_values = set(counts.index[:top_n])
    return series.apply(lambda x: x if x in top_values else "Other")

print("Reducing cardinality for high-dimensional features...")
df['path_feature'] = reduce_cardinality(df['path_feature'], 100)
df['browser'] = reduce_cardinality(df['browser'], 150)
df['city'] = reduce_cardinality(df['city'], 50)
df['state'] = reduce_cardinality(df['state'], 50)

# ==============================================================================
# CORE ALGORITHM 1: Pivot & Cluster
# ==============================================================================
def find_clean_distribution(subset_data, target_col, bucket_cols, threshold=0.15):
    """
    Input: A subset of data (e.g., only traffic from 'United States').
    Output: The 'Clean' probability distribution for that subset.
    """
    # 1. Create Buckets (e.g. "Week1-Chrome")
    # We use a copy to avoid SettingWithCopy warnings
    subset = subset_data.copy()
    subset['bucket'] = subset[bucket_cols].astype(str).agg('-'.join, axis=1)

    # Filter: Ignore tiny buckets (noise)
    bucket_counts = subset['bucket'].value_counts()
    valid_buckets = bucket_counts[bucket_counts > 20].index # Min 20 requests per bucket
    valid_data = subset[subset['bucket'].isin(valid_buckets)]

    if valid_data.empty:
        return None

    # 2. Pivot Matrix (Rows=Target Values, Cols=Buckets)
    pivot = pd.crosstab(valid_data[target_col], valid_data['bucket'])
    
    # Normalize to probabilities (Matrix Omega)
    distributions = pivot.div(pivot.sum(axis=0), axis=1)
    
    # 3. Cluster (Find Co-linear Columns using Jensen-Shannon)
    cols = distributions.columns
    n_cols = len(cols)
    if n_cols < 2: return distributions.mean(axis=1) # Fallback

    neighbor_counts = {}
    
    # Compare every column against every other column
    for i in range(n_cols):
        col_name = cols[i]
        ref_dist = distributions[col_name]
        neighbors = []
        
        for j in range(n_cols):
            if i == j: continue
            
            # Jensen-Shannon Distance
            try:
                dist = jensenshannon(ref_dist, distributions[cols[j]])
                if np.isnan(dist): dist = 1.0 
            except:
                dist = 1.0
                
            if dist < threshold:
                neighbors.append(cols[j])
        
        neighbor_counts[col_name] = neighbors

    # 4. Identify Best Cluster
    if not neighbor_counts:
        return distributions.mean(axis=1)

    best_center = max(neighbor_counts, key=lambda k: len(neighbor_counts[k]))
    best_cluster_cols = neighbor_counts[best_center] + [best_center]
    
    # 5. Average and Return
    clean_matrix = distributions[best_cluster_cols]
    estimated_dist = clean_matrix.mean(axis=1)
    
    return estimated_dist / estimated_dist.sum()

# ==============================================================================
# WRAPPER: Iterate Over Conditions
# ==============================================================================
def run_training(target, condition_feature, independent_features):
    print(f"\n--- Training Target: {target.upper()} (Condition: {condition_feature}) ---")
    
    results = {}
    
    # Get list of unique conditions (e.g., ['US', 'UK', 'CA'] or ['Chrome', 'Firefox'])
    conditions = df[condition_feature].unique()
    
    for cond_val in conditions:
        # Get Subset S_i (e.g., only US traffic)
        subset = df[df[condition_feature] == cond_val]
        
        # Skip tiny subsets
        if len(subset) < 50: 
            continue
            
        print(f"  > Analyzing '{cond_val}' ({len(subset)} rows)...", end=" ")
        
        # Run Algo 1 on this specific subset
        dist = find_clean_distribution(subset, target, independent_features)
        
        if dist is not None:
            results[cond_val] = dist
            print(f"Success ({len(dist)} items).")
        else:
            print("Skipped (not enough data).")

    # Save to CSV
    if results:
        # Combine into one DataFrame: Cols = Conditions, Rows = Target Values
        final_df = pd.DataFrame(results)
        # Fill missing values with 0 (if a status code exists in US but not UK)
        final_df.fillna(0, inplace=True)
        
        filename = os.path.join(OUTPUT_DIR, f"clean_{target}_by_{condition_feature}.csv")
        final_df.to_csv(filename)
        print(f"  [SAVED] {filename}")
    else:
        print(f"  [WARNING] No clean distributions found for {target}.")

# ==============================================================================
# EXECUTION: Replicating Section 6 Relations
# ==============================================================================

# 1. BROWSER VERSION
# Condition: Family (Chrome vs Firefox)
# Buckets: City, Status, etc.
run_training(
    target='browser', 
    condition_feature='family', 
    independent_features=['country', 'city', 'path_feature', 'status']
)

# 2. STATUS CODE
# Condition: Country
# Buckets: Week, Browser, Family
run_training(
    target='status',
    condition_feature='country',
    independent_features=['browser', 'week', 'family', 'osFamily']
)

# 3. PATH
# Condition: Country
run_training(
    target='path_feature',
    condition_feature='country',
    independent_features=['browser', 'week', 'family', 'osFamily']
)

# 4. BROWSER FAMILY
# Condition: Country
# Buckets: Path, Status
run_training(
    target='family',
    condition_feature='country',
    independent_features=['city', 'state', 'path_feature', 'status']
)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"All clean distributions saved in: '{OUTPUT_DIR}'")
print("="*60)