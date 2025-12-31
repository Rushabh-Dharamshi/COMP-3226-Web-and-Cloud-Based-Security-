import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "output_combined.csv")

class PROSDetector:
    """
    Comprehensive PROS implementation based on Herley's paper.
    Handles categorical features, conditional independence, and returns interpretable rules.
    """
    
    def __init__(self, min_samples_per_bin=10, jsd_threshold=0.15):
        self.min_samples_per_bin = min_samples_per_bin
        self.jsd_threshold = jsd_threshold
        self.clean_distributions = {}
        self.unattacked_bins = {}
        self.conditional_independence_rules = {}
        self.feature_stats = {}
        self._observed_probs = {}
        
    def parse_timestamp(self, timestamp_str):
        """Safely parse timestamp with various formats."""
        if pd.isna(timestamp_str):
            return pd.NaT
        
        # Try multiple formats
        try:
            # Try ISO format with Z
            return pd.to_datetime(timestamp_str, utc=True)
        except:
            try:
                # Try without timezone
                return pd.to_datetime(timestamp_str)
            except:
                return pd.NaT
    
    def engineer_features(self, df):
        """Create categorical features needed for PROS analysis."""
        df_engineered = df.copy()
        # --- ROBUST COLUMN FIX ---
        # If 'videoId' exists, rename it to 'videoID' so the rest of the code works
        if 'videoId' in df_engineered.columns:
            df_engineered.rename(columns={'videoId': 'videoID'}, inplace=True)
            
        # Optional: Handle 'channelId' vs 'channelID' just in case
        if 'channelId' in df_engineered.columns:
            df_engineered.rename(columns={'channelId': 'channelID'}, inplace=True)
        # -------------------------
        
        print("Engineering features...")
        
        # Safely convert timestamps
        df_engineered['commentDate'] = df_engineered['commentDate'].apply(self.parse_timestamp)
        df_engineered['channelDate'] = df_engineered['channelDate'].apply(self.parse_timestamp)
        
        # 1. Temporal Features
        print("  Creating temporal features...")
        df_engineered['comment_hour'] = df_engineered['commentDate'].dt.hour
        df_engineered['comment_dayofweek'] = df_engineered['commentDate'].dt.dayofweek
        
        # Create time bins
        df_engineered['time_of_day'] = pd.cut(
            df_engineered['comment_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
            include_lowest=True
        )
        
        # 2. Account Age Features
        print("  Creating account age features...")
        df_engineered['account_age_days'] = (
            (df_engineered['commentDate'] - df_engineered['channelDate']).dt.total_seconds() / (24 * 3600)
        ).fillna(0).clip(lower=0)
        
        df_engineered['account_age_bin'] = pd.cut(
            df_engineered['account_age_days'],
            bins=[-1, 1, 7, 30, 90, 365, float('inf')],
            labels=['<1d', '1-7d', '1w-1m', '1-3m', '3m-1y', '>1y']
        )
        
        # 3. Engagement Features
        print("  Creating engagement features...")
        # Convert to numeric safely
        df_engineered['commentLikeCount'] = pd.to_numeric(df_engineered['commentLikeCount'], errors='coerce')
        df_engineered['has_likes'] = (df_engineered['commentLikeCount'] > 0).astype(int)
        
        # Like count bins
        df_engineered['likes_bin'] = pd.cut(
            df_engineered['commentLikeCount'].fillna(0),
            bins=[-1, 0, 1, 5, 10, 50, float('inf')],
            labels=['0', '1', '2-5', '6-10', '11-50', '>50']
        )
        
        # 4. Channel Features
        print("  Creating channel features...")
        # Convert to numeric safely
        numeric_cols = ['channelSubscriberCount', 'channelVideoCount', 'channelViewCount']
        for col in numeric_cols:
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
        
        # Subscriber bins
        df_engineered['subscriber_bin'] = pd.cut(
            df_engineered['channelSubscriberCount'].fillna(0),
            bins=[-1, 0, 100, 1000, 10000, 100000, float('inf')],
            labels=['0', '1-100', '101-1k', '1k-10k', '10k-100k', '>100k']
        )
        
        # Video count bins
        df_engineered['video_count_bin'] = pd.cut(
            df_engineered['channelVideoCount'].fillna(0),
            bins=[-1, 0, 10, 50, 100, 500, float('inf')],
            labels=['0', '1-10', '11-50', '51-100', '101-500', '>500']
        )
        
        # 5. Behavioral Features
        print("  Creating behavioral features...")
        # Check for duplicate comments (same text from same channel)
        df_engineered['is_duplicate'] = 0
        if 'commentText' in df_engineered.columns:
            # Simple duplicate detection based on exact text match
            text_counts = df_engineered.groupby(['channelID', 'commentText']).size().reset_index(name='text_count')
            df_engineered = df_engineered.merge(
                text_counts[['channelID', 'commentText', 'text_count']],
                on=['channelID', 'commentText'],
                how='left'
            )
            df_engineered['is_duplicate'] = (df_engineered['text_count'] > 1).astype(int)
        
        # 6. Profile Features
        print("  Creating profile features...")
        df_engineered['has_country'] = df_engineered['channelCountry'].notna().astype(int)
        df_engineered['has_description'] = df_engineered['hasDescription'].fillna(False).astype(int)
        df_engineered['has_default_pic'] = df_engineered['defaultProfilePic'].fillna(True).astype(int)
        
        # Profile completeness score
        df_engineered['profile_complete_score'] = (
            df_engineered['has_country'] + 
            df_engineered['has_description'] + 
            (1 - df_engineered['has_default_pic'])
        )
        
        df_engineered['profile_complete_bin'] = pd.cut(
            df_engineered['profile_complete_score'],
            bins=[-1, 0, 1, 2, 3],
            labels=['None', 'Low', 'Medium', 'High']
        )
        
        # 7. Velocity Features (Posting frequency - RQ2)
        print("  Creating velocity features...")
        # Count comments per channel per video
        comment_counts = df_engineered.groupby(['channelID', 'videoID']).size().reset_index(name='comments_per_video')
        df_engineered = df_engineered.merge(comment_counts, on=['channelID', 'videoID'], how='left')
        
        df_engineered['comments_per_video_bin'] = pd.cut(
            df_engineered['comments_per_video'].fillna(1),
            bins=[0, 1, 3, 5, 10, float('inf')],
            labels=['1', '2-3', '4-5', '6-10', '>10']
        )
        
        # 8. Video Genre Features (RQ4)
        if 'videoGenre' in df_engineered.columns:
            print("  Creating genre features...")
            # Keep only top N genres, group others
            top_genres = df_engineered['videoGenre'].value_counts().head(10).index
            df_engineered['genre_group'] = df_engineered['videoGenre'].apply(
                lambda x: x if x in top_genres else 'Other'
            )
        else:
            df_engineered['genre_group'] = 'Unknown'
        
        print(f"Feature engineering complete. Created {len(df_engineered.columns)} features.")
        print(f"Sample features: {list(df_engineered.columns)[-15:]}")
        
        return df_engineered
    
    def define_conditional_independence(self, df):
        """
        Define conditional independence relations based on YouTube domain knowledge.
        Addresses RQ2: which features are conditionally independent.
        """
        print("\nDefining conditional independence rules...")
        
        # Based on YouTube comment analysis, we hypothesize:
        # 1. Within a time of day, account age is independent of subscriber count
        # 2. Within a subscriber bin, duplicate behavior is independent of video genre
        # 3. Within account age, profile completeness is independent of likes received
        
        self.conditional_independence_rules = {
            'account_age_bin': ['time_of_day', 'subscriber_bin'],
            'subscriber_bin': ['genre_group', 'comments_per_video_bin'],
            'profile_complete_bin': ['account_age_bin', 'likes_bin'],
            'time_of_day': ['subscriber_bin', 'video_count_bin'],
            'is_duplicate': ['genre_group', 'account_age_bin'],
            'comments_per_video_bin': ['genre_group', 'subscriber_bin']
        }
        
        print("Conditional independence rules defined:")
        for feature, independent_features in self.conditional_independence_rules.items():
            if feature in df.columns:
                print(f"  {feature} ⊥ {independent_features}")
        
        return self.conditional_independence_rules
    
    def find_unattacked_bins(self, df, target, independents):
        distributions = []
        
        # Algorithm 1: Pivot & Collect Distributions
        for feature in independents:
            if feature not in df.columns: continue
            
            for bin_val in df[feature].unique():
                subset = df[df[feature] == bin_val]
                if len(subset) < self.min_samples_per_bin: continue
                
                dist = subset[target].value_counts(normalize=True)
                distributions.append({
                    'feature': feature, 'bin': bin_val, 'distribution': dist, 'count': len(subset)
                })
        
        if not distributions: return None
        
        # Algorithm 1: Cluster Similar Distributions using JSD
        similar_groups = []
        visited = set()
        
        for i, d1 in enumerate(distributions):
            if i in visited: continue
            group = [d1]
            visited.add(i)
            
            for j, d2 in enumerate(distributions[i+1:], i+1):
                if j in visited: continue
                if self.are_distributions_similar(d1, d2):
                    group.append(d2)
                    visited.add(j)
            
            if len(group) > 1: similar_groups.append(group)
            
        if similar_groups:
            # Pick largest cluster as "Clean"
            best_group = max(similar_groups, key=len)
            
            # Average the distributions
            all_dists = [d['distribution'] for d in best_group]
            combined = pd.concat(all_dists, axis=1).fillna(0).mean(axis=1)
            combined = combined / combined.sum()
            
            self.clean_distributions[target] = combined.to_dict()
            self.unattacked_bins[target] = len(best_group)
            print(f"  Found clean distribution for '{target}' (based on {len(best_group)} bins)")
            return combined
            
        return None
    
    def are_distributions_similar(self, dist1, dist2):
        """Check if two distributions are similar."""
        try:
            # Align indices
            all_indices = dist1['distribution'].index.union(dist2['distribution'].index)
            p = dist1['distribution'].reindex(all_indices, fill_value=1e-10)
            q = dist2['distribution'].reindex(all_indices, fill_value=1e-10)
            
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            
            # JSD handles zeros gracefully
            dist = jensenshannon(p, q)
            
            if np.isnan(dist): return False
            return dist < self.jsd_threshold
        except:
            return False
    
    def estimate_clean_distributions(self, df):
        print("\n" + "="*40 + "\n[Algorithm 1] Estimating Clean Distributions\n" + "="*40)
        for target, independents in self.conditional_independence_rules.items():
            if target in df.columns:
                self.find_unattacked_bins(df, target, independents)

    def calculate_anomaly_scores(self, df):
        """
        [Algorithm 2] Joint Tuple Scoring.
        Calculates P_obs(Tuple) / P_clean(Tuple).
        """
        print("\n" + "="*40 + "\n[Algorithm 2] Calculating Joint Scores\n" + "="*40)
        df_scored = df.copy()
        
        # 1. Define Scoring Tuple
        possible_features = [
            'account_age_bin', 
            'subscriber_bin', 
            'genre_group', 
            'velocity_bin', 
            'profile_complete_bin', 
            'is_duplicate'
        ]
        
        tuple_cols = [f for f in possible_features if f in self.clean_distributions]
        
        if len(tuple_cols) < 2:
            print("Error: Not enough clean features for joint scoring.")
            df_scored['pros_anomaly_score'] = 0.0
            return df_scored
            
        print(f"Scoring Tuple ({len(tuple_cols)} features): {tuple_cols}")
        
        # 2. Calculate P_Clean (Joint)
        print("Calculating Joint Clean Probability...")
        
        def get_clean_prob(feature, val):
            # Safe lookup
            return self.clean_distributions[feature].get(str(val), 1e-9)

        # Initialize as pure float
        df_scored['P_clean_joint'] = 1.0
        
        for col in tuple_cols:
            # FIX IS HERE: .astype(float) forces it to be a number, not a category
            # Also cast input 'x' to str() to match dictionary keys
            df_scored[f'p_clean_{col}'] = df_scored[col].apply(lambda x: get_clean_prob(col, x)).astype(float)
            
            # Now multiplication will work because both sides are definitely floats
            df_scored['P_clean_joint'] *= df_scored[f'p_clean_{col}']
            
        # 3. Calculate P_Observed (Joint)
        print("Calculating Joint Observed Frequency...")
        
        counts = df_scored.groupby(tuple_cols).size().reset_index(name='tuple_count')
        total_rows = len(df_scored)
        counts['P_obs_joint'] = counts['tuple_count'] / total_rows
        
        df_scored = df_scored.merge(counts, on=tuple_cols, how='left')
        
        # 4. Calculate Odds Score
        print("Computing Final Bot Odds...")
        alpha = 0.5
        
        # Ensure division happens on floats
        df_scored['P_obs_joint'] = df_scored['P_obs_joint'].astype(float)
        
        df_scored['pros_anomaly_score'] = (df_scored['P_obs_joint'] / (alpha * df_scored['P_clean_joint'])) - 1
        df_scored['pros_anomaly_score'] = df_scored['pros_anomaly_score'].clip(lower=0)
        
        df_scored.sort_values('pros_anomaly_score', ascending=False, inplace=True)
        return df_scored
    
    def generate_interpretable_rules(self, df_scored, top_n=20):
        """
        Generate human-interpretable rules for top anomalies.
        Addresses RQ3: interpretable PROS technique.
        """
        print("\n" + "=" * 60)
        print("Generating Interpretable Rules (RQ3)")
        print("=" * 60)
        
        # Group by channel for analysis
        channel_features = [
            'channelID', 'channelTitle', 'videoGenre', 'genre_group',
            'account_age_bin', 'subscriber_bin', 'profile_complete_bin',
            'time_of_day', 'is_duplicate', 'comments_per_video_bin'
        ]
        
        # Only use columns that exist
        available_features = [f for f in channel_features if f in df_scored.columns]
        
        channel_scores = df_scored.groupby('channelID').agg({
            'pros_anomaly_score': 'max', # Use max for Joint Odds
            'channelTitle': 'first',
            'videoGenre': 'first',
            'is_duplicate': 'mean'
        }).reset_index().sort_values('pros_anomaly_score', ascending=False)
        
        channel_scores = channel_scores.sort_values('pros_anomaly_score', ascending=False)
        
        rules = []
        for idx, row in channel_scores.head(top_n).iterrows():
            rules.append({
                'channel': str(row['channelTitle']),
                'genre': str(row['videoGenre']),
                'score': row['pros_anomaly_score'],
                'rule': f"Joint Probability Anomaly (Score: {row['pros_anomaly_score']:.2e})"
            })
        return rules
    
    def analyze_channel_categories(self, df_scored):
        print("\n" + "=" * 60)
        print("Analyzing Channel Categories (RQ4)")
        print("=" * 60)
        
        genre_col = 'genre_group' if 'genre_group' in df_scored.columns else 'videoGenre'
        if genre_col not in df_scored.columns: return None
        
        # 1. Calculate the rich statistics
        genre_stats = df_scored.groupby(genre_col).agg({
            'pros_anomaly_score': ['mean', 'std', 'count'],
            'is_duplicate': 'mean',
            'channelID': 'nunique'
        }).round(3)
        
        # 2. Flatten the column names (This creates 'pros_anomaly_score_mean')
        genre_stats.columns = ['_'.join(col).strip('_') for col in genre_stats.columns.values]
        genre_stats = genre_stats.reset_index().sort_values('pros_anomaly_score_mean', ascending=False)
        
        # 3. Print the results
        print(f"\nAnalyzed {len(genre_stats)} categories:")
        for idx, row in genre_stats.head().iterrows():
            print(f"  {row[genre_col]}: avg_score={row['pros_anomaly_score_mean']:.3f}")
            
        # 4. FIX: Return the 'genre_stats' DF, which contains 'pros_anomaly_score_mean'
        return genre_stats
    
    
    def compare_with_isolation_forest(self, df):
        """Compare PROS with Isolation Forest baseline."""
        try:
            from sklearn.ensemble import IsolationForest
            
            print("\nComparing with Isolation Forest...")
            
            # Select features for Isolation Forest
            categorical_features = [
                'account_age_bin', 'subscriber_bin', 'video_count_bin',
                'time_of_day', 'profile_complete_bin'
            ]
            
            # Only use features that exist
            available_features = [f for f in categorical_features if f in df.columns]
            
            if not available_features:
                print("No features available for Isolation Forest")
                return df
            
            # Prepare data for Isolation Forest - handle categorical columns properly
            features_df = df[available_features].copy()
            
            # For each categorical column, handle missing values properly
            for col in available_features:
                if pd.api.types.is_categorical_dtype(features_df[col]):
                    # For categorical columns, add 'Missing' as a valid category first
                    features_df[col] = features_df[col].cat.add_categories(['Missing'])
                    features_df[col] = features_df[col].fillna('Missing')
                else:
                    # For non-categorical, just convert to string
                    features_df[col] = features_df[col].astype(str).fillna('Missing')
            
            # One-hot encode
            features_df = pd.get_dummies(
                features_df,
                columns=available_features,
                drop_first=True
            )
            
            # Handle empty dataframe
            if features_df.shape[1] == 0:
                print("No features after encoding")
                return df
            
            # Check if we have enough data for Isolation Forest
            if len(features_df) < 10:
                print(f"Not enough data for Isolation Forest (only {len(features_df)} samples)")
                return df
            
            # Train Isolation Forest
            # Estimate contamination based on PROS scores if available
            contamination = 0.1  # Assume top 10% are bots (same assumption we make for PROS)
            print(f"  Using contamination rate: {contamination:.1%}")
            
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
            
            iso_scores = iso_forest.fit_predict(features_df)
            
            # Convert to anomaly scores (1 = normal, -1 = anomaly)
            df['iso_forest_score'] = iso_scores
            df['iso_forest_anomaly'] = (iso_scores == -1).astype(int)
            
            # Calculate similarity with PROS
            if 'pros_anomaly_score' in df.columns:
                # Create binary classification from PROS scores (top 10% as anomalies)
                pros_threshold = df['pros_anomaly_score'].quantile(0.9)
                df['pros_anomaly_binary'] = (df['pros_anomaly_score'] > pros_threshold).astype(int)
                
                # Calculate agreement
                agreement = (df['iso_forest_anomaly'] == df['pros_anomaly_binary']).mean()
                print(f"  Agreement between PROS and Isolation Forest: {agreement:.2%}")
                
                # Show how many flagged by each method
                pros_flags = df['pros_anomaly_binary'].sum()
                iso_flags = df['iso_forest_anomaly'].sum()
                print(f"  PROS flagged: {pros_flags} accounts ({pros_flags/len(df):.1%})")
                print(f"  Isolation Forest flagged: {iso_flags} accounts ({iso_flags/len(df):.1%})")
                
                # Calculate precision and recall if we have some ground truth
                # (For demonstration, we'll use PROS as "pseudo-ground truth")
                true_positives = ((df['pros_anomaly_binary'] == 1) & (df['iso_forest_anomaly'] == 1)).sum()
                false_positives = ((df['pros_anomaly_binary'] == 0) & (df['iso_forest_anomaly'] == 1)).sum()
                false_negatives = ((df['pros_anomaly_binary'] == 1) & (df['iso_forest_anomaly'] == 0)).sum()
                
                if (true_positives + false_positives) > 0:
                    precision = true_positives / (true_positives + false_positives)
                    print(f"  Precision (vs PROS): {precision:.2%}")
                
                if (true_positives + false_negatives) > 0:
                    recall = true_positives / (true_positives + false_negatives)
                    print(f"  Recall (vs PROS): {recall:.2%}")
            
            return df
            
        except Exception as e:
            print(f"Error running Isolation Forest: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    
    
    def run_full_analysis(self, df):
        """Run complete PROS analysis pipeline."""
        print("=" * 60)
        print("PROS BOT DETECTION ANALYSIS (Joint Tuple + JSD)")
        print("=" * 60)
        
        # Step 1: Feature Engineering
        df_engineered = self.engineer_features(df)
        
        # Step 2: Define Rules
        self.define_conditional_independence(df_engineered)
        
        # Step 3: Estimate Clean Distributions (Algorithm 1)
        self.estimate_clean_distributions(df_engineered)
        
        # Step 4: Calculate Scores (Algorithm 2)
        df_scored = self.calculate_anomaly_scores(df_engineered)
        
        # Step 5: Rules (RQ3)
        rules = self.generate_interpretable_rules(df_scored)
        
        # Step 6: Categories (RQ4)
        genre_analysis = self.analyze_channel_categories(df_scored)
        
        # Step 7: Comparison
        # Note: IsoForest comparison might take a long time on millions of rows.
        # You might want to sample if it's too slow.
        df_comparison = self.compare_with_isolation_forest(df_scored.sample(min(len(df_scored), 100000)))
        
        return {
            'df_scored': df_scored,
            'rules': rules,
            'genre_analysis': genre_analysis,
            'clean_distributions': self.clean_distributions,
            'unattacked_bins': self.unattacked_bins,
            'feature_stats': self.feature_stats,
            'df_comparison': df_comparison
        }
def visualize_results(results):
    """Create visualizations for the analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df_scored = results['df_scored']
        genre_analysis = results['genre_analysis']
        rules = results['rules']
        feature_stats = results.get('feature_stats', {})
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Distribution of PROS scores
        plt.figure(figsize=(14, 10))
        
        # Plot 1: PROS score distribution
        plt.subplot(2, 2, 1)
        if 'pros_anomaly_score' in df_scored.columns:
            sns.histplot(df_scored['pros_anomaly_score'], bins=50, kde=True, color='steelblue')
            plt.axvline(x=0, color='red', linestyle='--', label='Neutral (0)', linewidth=2)
            plt.axvline(x=df_scored['pros_anomaly_score'].quantile(0.9), 
                       color='orange', linestyle=':', label='Top 10%', linewidth=2)
            plt.xlabel('PROS Anomaly Score', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('Distribution of PROS Anomaly Scores', fontsize=14, fontweight='bold')
            plt.legend()
        
        # Plot 2: Top genres by anomaly score (RQ4)
        if genre_analysis is not None and len(genre_analysis) > 0:
            plt.subplot(2, 2, 2)
            top_genres = genre_analysis.head(8)
            genre_col = genre_analysis.columns[0]  # First column is genre name
            bars = plt.barh(top_genres[genre_col], top_genres['pros_anomaly_score_mean'])
            
            # Color bars based on score
            for bar, score in zip(bars, top_genres['pros_anomaly_score_mean']):
                if score > 0.5:
                    bar.set_color('firebrick')
                elif score > 0:
                    bar.set_color('darkorange')
                else:
                    bar.set_color('steelblue')
            
            plt.xlabel('Average Anomaly Score', fontsize=12)
            plt.title('Top Genres by Bot Likelihood (RQ4)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
        
        # Plot 3: Feature importance (RQ2)
        if feature_stats:
            plt.subplot(2, 2, 3)
            features = list(feature_stats.keys())
            num_bins = [stats['num_unattacked_bins'] for stats in feature_stats.values()]
            
            if features and num_bins:
                # Create horizontal bar chart
                y_pos = range(len(features))
                bars = plt.barh(y_pos, num_bins)
                
                # Color bars
                for i, (feature, stats) in enumerate(feature_stats.items()):
                    if stats['num_unattacked_bins'] > 5:
                        bars[i].set_color('seagreen')
                    elif stats['num_unattacked_bins'] > 2:
                        bars[i].set_color('goldenrod')
                    else:
                        bars[i].set_color('lightcoral')
                
                plt.yticks(y_pos, features)
                plt.xlabel('Number of Unattacked Bins Found')
                plt.title('Feature Importance for Detection (RQ2)', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
        
        # Plot 4: Comparison of features for top vs normal channels
        plt.subplot(2, 2, 4)
        if 'pros_anomaly_score' in df_scored.columns and 'account_age_days' in df_scored.columns:
            # Identify top 10% as suspicious
            threshold = df_scored['pros_anomaly_score'].quantile(0.9)
            df_scored['is_suspicious'] = df_scored['pros_anomaly_score'] > threshold
            
            # Compare account age
            suspicious_age = df_scored[df_scored['is_suspicious']]['account_age_days'].clip(upper=365)
            normal_age = df_scored[~df_scored['is_suspicious']]['account_age_days'].clip(upper=365)
            
            box_data = [normal_age.dropna(), suspicious_age.dropna()]
            box = plt.boxplot(box_data, labels=['Normal', 'Suspicious'], patch_artist=True)
            
            # Color boxes
            box['boxes'][0].set_facecolor('lightblue')
            box['boxes'][1].set_facecolor('lightcoral')
            
            plt.ylabel('Account Age (days, capped at 365)', fontsize=11)
            plt.title('Account Age: Normal vs Suspicious Channels', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Print interpretable rules
        print("\n" + "=" * 80)
        print("TOP INTERPRETABLE RULES (RQ3)")
        print("=" * 80)
        
        if rules:
            print(f"\nGenerated {len(rules)} interpretable rules:")
            print("-" * 80)
            
            for i, rule in enumerate(rules[:10], 1):
                print(f"\nRule #{i}:")
                print(f"  Channel: {rule['channel'][:50]}...")
                print(f"  Genre: {rule['genre']}")
                print(f"  Bot Score: {rule['score']:.3f}")
                print(f"  Detection Rule: {rule['rule']}")
                
                # Show feature breakdown
                if 'features' in rule:
                    print(f"  Key Indicators: {', '.join(rule['features'])}")
        else:
            print("No interpretable rules generated.")
        
        # 3. Print summary statistics
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        df_scored = results['df_scored']
        print(f"\nDataset Statistics:")
        print(f"  Total comments analyzed: {len(df_scored):,}")
        print(f"  Unique channels: {df_scored['channelID'].nunique():,}")
        
        if 'pros_anomaly_score' in df_scored.columns:
            print(f"\nPROS Score Statistics:")
            print(f"  Average score: {df_scored['pros_anomaly_score'].mean():.3f}")
            print(f"  Standard deviation: {df_scored['pros_anomaly_score'].std():.3f}")
            print(f"  Minimum score: {df_scored['pros_anomaly_score'].min():.3f}")
            print(f"  Maximum score: {df_scored['pros_anomaly_score'].max():.3f}")
            
            # Identify suspicious channels
            threshold = df_scored['pros_anomaly_score'].quantile(0.9)
            suspicious = df_scored[df_scored['pros_anomaly_score'] > threshold]
            print(f"\nSuspicious Activity Detection:")
            print(f"  Channels flagged as suspicious (top 10%): {suspicious['channelID'].nunique():,}")
            print(f"  Percentage of total: {suspicious['channelID'].nunique()/df_scored['channelID'].nunique():.1%}")
            
            if 'genre_group' in suspicious.columns:
                top_suspicious_genres = suspicious['genre_group'].value_counts().head(3)
                print(f"  Top suspicious genres: {', '.join([f'{g} ({c})' for g, c in top_suspicious_genres.items()])}")
        
        # 4. Print feature statistics
        if results.get('feature_stats'):
            print(f"\nFeature Analysis (RQ2):")
            print("-" * 40)
            for feature, stats in results['feature_stats'].items():
                print(f"  {feature}:")
                print(f"    Unattacked bins found: {stats['num_unattacked_bins']}")
                if 'top_values' in stats:
                    top_vals = list(stats['top_values'].items())[:2]
                    print(f"    Top values in clean dist: {', '.join([f'{k}: {v:.3f}' for k, v in top_vals])}")
        
    except ImportError:
        print("Matplotlib or seaborn not installed. Skipping visualizations.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

# ---------------- MAIN EXECUTION ----------------

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please run the data fetching script first.")
    else:
        # Load data
        print(f"Loading data from {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)
        
        if df.empty:
            print("CSV file is empty.")
        else:
            print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            
            # Initialize and run PROS detector
            detector = PROSDetector(min_samples_per_bin=5,jsd_threshold=0.15)
            
            # Run full analysis
            results = detector.run_full_analysis(df)
            
            # Save results
            output_path = os.path.join(BASE_DIR, "pros_results.csv")
            results['df_scored'].to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            
            # Display results
            visualize_results(results)