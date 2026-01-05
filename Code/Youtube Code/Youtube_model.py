import pandas as pd
import numpy as np
import os
import json
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Youtube Data")
CSV_PATH = os.path.join(DATA_DIR, "Youtube_extracted_data.csv")

YOUTUBE_RESULTS_DIR = os.path.join(PROJECT_DIR, "Youtube Results")
os.makedirs(YOUTUBE_RESULTS_DIR, exist_ok=True)

PLOT_OUTPUT_PATH = os.path.join(YOUTUBE_RESULTS_DIR, "pros_results_visualization.png")


class PROSDetector:
    """
    PROS (Pivot and Seek Rank-One Submatrix) Bot Detection System
    
    This implementation directly addresses the research questions from your mid-term report:
    
    RQ1: How can unsupervised anomaly detection identify automated commenting behaviour 
         on YouTube without relying on labelled data?
         → Implemented via clean distribution estimation (Algorithm 1) and anomaly scoring (Algorithm 2)
    
    RQ2: Which structural and behavioural features most effectively differentiate bots from human users?
         → Analyzed through conditional independence rules and feature importance metrics
    
    RQ3: How can the PROS technique be applied while remaining interpretable to human analysts?
         → Achieved through human-readable rules and transparent scoring mechanisms
    
    RQ4: What YouTube channel categories and genres have the most bot traffic?
         → Answered via genre-specific bot likelihood analysis
    """
    
    def __init__(self, min_samples_per_bin=10, jsd_threshold=0.15):
        self.min_samples_per_bin = min_samples_per_bin
        self.jsd_threshold = jsd_threshold
        self.clean_distributions = {}
        self.unattacked_bins = {}
        self.conditional_independence_rules = {}
        self.feature_stats = {}
        self._observed_probs = {}
        
        # Track RQ-specific metrics
        self.rq_metrics = {
            'rq1_unsupervised': {},
            'rq2_feature_importance': {},
            'rq3_interpretability': {},
            'rq4_genre_analysis': {}
        }

    def parse_timestamp(self, timestamp_str):
        """Safely parse timestamp with various formats."""
        if pd.isna(timestamp_str):
            return pd.NaT
        try:
            return pd.to_datetime(timestamp_str, utc=True)
        except Exception:
            try:
                return pd.to_datetime(timestamp_str)
            except Exception:
                return pd.NaT

    def engineer_features(self, df):
        """
        Create categorical features needed for PROS analysis.
        Addresses RQ2: Feature engineering for bot differentiation.
        """
        df_engineered = df.copy()

        # Column normalisation
        if 'videoId' in df_engineered.columns:
            df_engineered.rename(columns={'videoId': 'videoID'}, inplace=True)
        if 'channelId' in df_engineered.columns:
            df_engineered.rename(columns={'channelId': 'channelID'}, inplace=True)

        print("\n" + "="*70)
        print("FEATURE ENGINEERING (RQ2: Structural & Behavioral Features)")
        print("="*70)

        # --- Timestamp handling (vectorized) ---
        print("\n[1] TEMPORAL FEATURES")
        df_engineered['commentDate'] = pd.to_datetime(
            df_engineered['commentDate'], errors='coerce', utc=True
        )
        df_engineered['channelDate'] = pd.to_datetime(
            df_engineered['channelDate'], errors='coerce', utc=True
        )

        df_engineered['comment_hour'] = df_engineered['commentDate'].dt.hour
        df_engineered['comment_dayofweek'] = df_engineered['commentDate'].dt.dayofweek

        df_engineered['time_of_day'] = pd.cut(
            df_engineered['comment_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
            include_lowest=True
        )
        print(f"    ✓ Created: comment_hour, comment_dayofweek, time_of_day")

        # --- Account age (CRITICAL for bot detection) ---
        print("\n[2] ACCOUNT AGE FEATURES (Key Bot Indicator)")
        age_delta = (df_engineered['commentDate'] - df_engineered['channelDate'])
        age_days = age_delta.dt.total_seconds().div(24 * 3600)
        df_engineered['account_age_days'] = age_days.fillna(0).clip(lower=0)

        df_engineered['account_age_bin'] = pd.cut(
            df_engineered['account_age_days'],
            bins=[-1, 1, 7, 30, 90, 365, float('inf')],
            labels=['<1d', '1-7d', '1w-1m', '1-3m', '3m-1y', '>1y']
        )
        
        # Track account age distribution for RQ2
        age_dist = df_engineered['account_age_bin'].value_counts(normalize=True).to_dict()
        print(f"    Created: account_age_days, account_age_bin")
        print(f"    Account Age Distribution:")
        for age_range, pct in sorted(age_dist.items(), key=lambda x: str(x[0])):
            print(f"       {age_range}: {pct:.1%}")

        # --- Engagement features ---
        print("\n[3] ENGAGEMENT FEATURES")
        df_engineered['commentLikeCount'] = pd.to_numeric(
            df_engineered['commentLikeCount'], errors='coerce'
        ).fillna(0)

        if 'likes_bin' in df_engineered.columns:
            df_engineered.drop(columns=['likes_bin'], inplace=True)

        df_engineered['has_likes'] = (df_engineered['commentLikeCount'] > 0).astype(np.int8)

        df_engineered['likes_bin'] = pd.cut(
            df_engineered['commentLikeCount'],
            bins=[-1, 0, 1, 5, 10, 50, float('inf')],
            labels=['0', '1', '2-5', '6-10', '11-50', '>50']
        )
        
        likes_pct = (df_engineered['has_likes'] == 1).mean()
        print(f"     Created: commentLikeCount, has_likes, likes_bin")
        print(f"     {likes_pct:.1%} of comments have likes")

        # --- Channel features (CRITICAL for maturity assessment) ---
        print("\n[4] CHANNEL MATURITY FEATURES (Novel Contribution)")
        numeric_cols = ['channelSubscriberCount', 'channelVideoCount', 'channelViewCount']
        for col in numeric_cols:
            if col in df_engineered.columns:
                df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
            else:
                df_engineered[col] = np.nan

        df_engineered['subscriber_bin'] = pd.cut(
            df_engineered['channelSubscriberCount'].fillna(0),
            bins=[-1, 0, 100, 1000, 10000, 100000, float('inf')],
            labels=['0', '1-100', '101-1k', '1k-10k', '10k-100k', '>100k']
        )

        df_engineered['video_count_bin'] = pd.cut(
            df_engineered['channelVideoCount'].fillna(0),
            bins=[-1, 0, 10, 50, 100, 500, float('inf')],
            labels=['0', '1-10', '11-50', '51-100', '101-500', '>500']
        )
        print(f"    ✓ Created: subscriber_bin, video_count_bin")

        # --- Behavioral features (duplicates) ---
        print("\n[5] BEHAVIORAL FEATURES (Coordination Detection)")
        df_engineered['is_duplicate'] = 0
        if 'commentText' in df_engineered.columns and 'channelID' in df_engineered.columns:
            text_counts = (
                df_engineered
                .groupby(['channelID', 'commentText'])['commentText']
                .transform('size')
            )
            df_engineered['is_duplicate'] = (text_counts > 1).astype(np.int8)
        
        dup_pct = (df_engineered['is_duplicate'] == 1).mean()
        print(f"     Created: is_duplicate")
        print(f"     {dup_pct:.1%} of comments are duplicates (possible coordination)")

        # --- Profile features ---
        print("\n[6] PROFILE COMPLETENESS FEATURES")
        df_engineered['has_country'] = df_engineered['channelCountry'].notna().astype(np.int8)
        df_engineered['has_description'] = df_engineered['hasDescription'].fillna(False).astype(np.int8)
        df_engineered['has_default_pic'] = df_engineered['defaultProfilePic'].fillna(True).astype(np.int8)

        profile_score = (
            df_engineered['has_country'].to_numpy()
            + df_engineered['has_description'].to_numpy()
            + (1 - df_engineered['has_default_pic'].to_numpy())
        )
        df_engineered['profile_complete_score'] = profile_score

        df_engineered['profile_complete_bin'] = pd.cut(
            df_engineered['profile_complete_score'],
            bins=[-1, 0, 1, 2, 3],
            labels=['None', 'Low', 'Medium', 'High']
        )
        
        profile_dist = df_engineered['profile_complete_bin'].value_counts(normalize=True).to_dict()
        print(f"     Created: profile_complete_score, profile_complete_bin")
        print(f"     Profile Completeness:")
        for level, pct in sorted(profile_dist.items(), key=lambda x: str(x[0])):
            print(f"       {level}: {pct:.1%}")

        # --- Velocity features (CRITICAL for bot mob detection) ---
        print("\n[7] VELOCITY FEATURES (Commenter Mob Detection)")
        if 'channelID' in df_engineered.columns and 'videoID' in df_engineered.columns:
            df_engineered['comments_per_video'] = (
                df_engineered
                .groupby(['channelID', 'videoID'])['videoID']
                .transform('size')
            )
        else:
            df_engineered['comments_per_video'] = 1

        df_engineered['comments_per_video_bin'] = pd.cut(
            df_engineered['comments_per_video'].fillna(1),
            bins=[0, 1, 3, 5, 10, float('inf')],
            labels=['1', '2-3', '4-5', '6-10', '>10']
        )
        
        velocity_dist = df_engineered['comments_per_video_bin'].value_counts(normalize=True).to_dict()
        high_velocity = df_engineered[df_engineered['comments_per_video'] > 10]['channelID'].nunique()
        print(f"     Created: comments_per_video, comments_per_video_bin")
        print(f"     {high_velocity:,} channels with >10 comments/video (high velocity)")

        # --- Genre features ---
        if 'videoGenre' in df_engineered.columns:
            print("\n[8] GENRE FEATURES (RQ4 Analysis)")
            top_genres = df_engineered['videoGenre'].value_counts().head(10).index
            df_engineered['genre_group'] = df_engineered['videoGenre'].where(
                df_engineered['videoGenre'].isin(top_genres), other='Other'
            )
            
            genre_dist = df_engineered['genre_group'].value_counts().head()
            print(f"     Created: genre_group")
            print(f"     Top Genres:")
            for genre, count in genre_dist.items():
                print(f"       {genre}: {count:,} comments")
        else:
            df_engineered['genre_group'] = 'Unknown'

        print(f"\n{'='*70}")
        print(f"✓ Feature engineering complete: {len(df_engineered.columns)} total features")
        print(f"{'='*70}")

        return df_engineered

    def define_conditional_independence(self, df):
        """
        Define conditional independence relations based on YouTube domain knowledge.
        Addresses RQ2: Which features are conditionally independent and useful for detection.
        """
        print("\n" + "="*70)
        print("CONDITIONAL INDEPENDENCE RULES (RQ2: Feature Relationships)")
        print("="*70)

        self.conditional_independence_rules = {
            'account_age_bin': ['time_of_day', 'subscriber_bin'],
            'subscriber_bin': ['genre_group', 'comments_per_video_bin'],
            'profile_complete_bin': ['account_age_bin', 'likes_bin'],
            'time_of_day': ['subscriber_bin', 'video_count_bin'],
            'is_duplicate': ['genre_group', 'account_age_bin'],
            'comments_per_video_bin': ['genre_group', 'subscriber_bin']
        }

        print("\nThese rules encode domain knowledge about YouTube behavior:")
        print("\n[Rationale]")
        for feature, independent_features in self.conditional_independence_rules.items():
            if feature in df.columns:
                print(f"\n  {feature} ⊥ {independent_features}")
                
                # Explain the rationale
                if feature == 'account_age_bin':
                    print(f"     → Account age shouldn't depend on time of day or channel size")
                elif feature == 'subscriber_bin':
                    print(f"     → Channel size shouldn't depend on genre or posting velocity")
                elif feature == 'time_of_day':
                    print(f"     → Posting time shouldn't depend on channel metrics")
                elif feature == 'is_duplicate':
                    print(f"     → Duplicate behavior shouldn't depend on genre or account age")
                elif feature == 'comments_per_video_bin':
                    print(f"     → Velocity shouldn't depend on genre or channel size")

        print(f"\n{'='*70}")
        return self.conditional_independence_rules

    def find_unattacked_bins(self, df, target, independents):
        """
        Find unattacked bins and estimate clean distribution for a target feature.
        Core of RQ1: Unsupervised detection without labels.
        """
        print(f"    Processing {target}...")
        distributions = []

        for feature in independents:
            if feature not in df.columns:
                continue

            grouped = df.groupby(feature)[target]
            sizes = grouped.size()
            valid_bins = sizes[sizes >= self.min_samples_per_bin].index
            if len(valid_bins) == 0:
                continue

            value_counts = (
                df[df[feature].isin(valid_bins)]
                .groupby(feature)[target]
                .value_counts(normalize=True)
                .rename('prob')
                .reset_index()
            )

            for bin_val in valid_bins:
                sub = value_counts[value_counts[feature] == bin_val]
                if sub.empty:
                    continue
                dist = pd.Series(sub['prob'].to_numpy(), index=sub[target].astype(str).to_numpy())
                distributions.append(
                    {
                        'feature': feature,
                        'bin': bin_val,
                        'distribution': dist,
                        'count': int(sizes.loc[bin_val]),
                    }
                )

        if not distributions:
            print(f"    ✗ No distributions found for {target}")
            return False

        print(f"    → Collected {len(distributions)} candidate distributions")

        # Cluster similar distributions using JSD (Jensen-Shannon Divergence)
        similar_groups = []
        visited = set()
        n = len(distributions)

        for i in range(n):
            if i in visited:
                continue
            d1 = distributions[i]
            group = [d1]
            visited.add(i)
            for j in range(i + 1, n):
                if j in visited:
                    continue
                d2 = distributions[j]
                if self.are_distributions_similar(d1, d2):
                    group.append(d2)
                    visited.add(j)
            if len(group) > 1:
                similar_groups.append(group)

        if similar_groups:
            best_group = max(similar_groups, key=len)

            print(f"    ✓ Found {len(best_group)} UNATTACKED bins (clean traffic cluster)")

            # Align all distributions
            all_indices = set()
            for d in best_group:
                all_indices.update(d['distribution'].index)
            all_indices = sorted(all_indices)

            aligned_arrays = []
            for d in best_group:
                dist = d['distribution'].reindex(all_indices, fill_value=0).to_numpy(dtype=float)
                s = dist.sum()
                if s > 0:
                    dist /= s
                aligned_arrays.append(dist)

            avg_array = np.mean(aligned_arrays, axis=0)
            s = avg_array.sum()
            if s > 0:
                avg_array /= s

            combined = pd.Series(avg_array, index=all_indices)

            self.clean_distributions[target] = {
                str(k): float(v) for k, v in combined.to_dict().items()
            }

            self.unattacked_bins[target] = len(best_group)
            self.feature_stats[target] = {
                'num_unattacked_bins': len(best_group),
                'top_values': dict(combined.sort_values(ascending=False).head(5)),
                'independent_features': independents
            }
            
            # Store for RQ2 analysis
            self.rq_metrics['rq2_feature_importance'][target] = {
                'unattacked_bins': len(best_group),
                'effectiveness_score': len(best_group) / len(distributions) if distributions else 0
            }
            
            return True
        else:
            print(f"    ✗ No similar distribution clusters found (attack too widespread)")
            return False

    def are_distributions_similar(self, dist1, dist2):
        """Check if two distributions are similar using Jensen-Shannon Divergence."""
        try:
            idx = dist1['distribution'].index.union(dist2['distribution'].index)
            p = dist1['distribution'].reindex(idx, fill_value=1e-10).to_numpy(dtype=float)
            q = dist2['distribution'].reindex(idx, fill_value=1e-10).to_numpy(dtype=float)

            p_sum = p.sum()
            q_sum = q.sum()
            if p_sum > 0:
                p /= p_sum
            if q_sum > 0:
                q /= q_sum

            dist = jensenshannon(p, q)
            if np.isnan(dist):
                return False
            return dist < self.jsd_threshold
        except Exception:
            return False

    def estimate_clean_distributions(self, df):
        """
        Algorithm 1: Estimate clean distributions for all features.
        Core implementation of RQ1: Unsupervised anomaly detection.
        """
        print("\n" + "="*70)
        print("ALGORITHM 1: ESTIMATING CLEAN DISTRIBUTIONS (RQ1)")
        print("="*70)
        print("\nObjective: Identify 'unattacked bins' to model normal behavior")
        print("Method: Find feature values with similar conditional distributions")
        print("="*70 + "\n")
        
        successful_targets = []

        for target, independents in self.conditional_independence_rules.items():
            if target not in df.columns:
                print(f"  Skipping {target} - not in DataFrame")
                continue

            success = self.find_unattacked_bins(df, target, independents)
            if success:
                successful_targets.append(target)

        print(f"\n{'='*70}")
        print(f"✓ RQ1 SUCCESS: Estimated clean distributions for {len(successful_targets)}/{len(self.conditional_independence_rules)} features")
        print(f"{'='*70}")
        
        if successful_targets:
            print("\n[Clean Distribution Summary]")
            for target in successful_targets:
                if target in self.clean_distributions:
                    n_values = len(self.clean_distributions[target])
                    n_bins = self.unattacked_bins.get(target, 0)
                    print(f"  • {target}: {n_values} categories, {n_bins} unattacked bins")
        
        # Store RQ1 metrics
        self.rq_metrics['rq1_unsupervised'] = {
            'total_features': len(self.conditional_independence_rules),
            'successful_features': len(successful_targets),
            'success_rate': len(successful_targets) / len(self.conditional_independence_rules) if self.conditional_independence_rules else 0,
            'features_used': successful_targets
        }

        return successful_targets

    def calculate_anomaly_scores(self, df):
        """
        Algorithm 2: Calculate anomaly scores using clean distributions.
        Implements the PROS detection mechanism for RQ1 and RQ3.
        """
        print("\n" + "="*70)
        print("ALGORITHM 2: CALCULATING ANOMALY SCORES (RQ1 + RQ3)")
        print("="*70)
        print("\nMethod: Compare observed joint probability vs. clean joint probability")
        print("Formula: Score = P_obs(Tuple) / (α × P_clean(Tuple)) - 1")
        print("="*70 + "\n")
        
        df_scored = df.copy()

        possible_features = [
            'account_age_bin',
            'subscriber_bin',
            'genre_group',
            'comments_per_video_bin',
            'profile_complete_bin',
            'is_duplicate'
        ]
        tuple_cols = [f for f in possible_features if f in self.clean_distributions]

        if len(tuple_cols) < 2:
            print("⚠ Warning: Not enough clean features for joint scoring.")
            df_scored['pros_anomaly_score'] = 0.0
            return df_scored

        print(f"[Scoring Features] Using {len(tuple_cols)}-tuple: {tuple_cols}\n")

        # Calculate clean probabilities
        clean_maps = {
            feature: self.clean_distributions[feature] for feature in tuple_cols
        }

        df_scored['P_clean_joint'] = 1.0

        for col in tuple_cols:
            col_as_str = df_scored[col].astype(str)
            probs = col_as_str.map(clean_maps[col]).fillna(1e-9).to_numpy(dtype=float)
            df_scored[f'p_clean_{col}'] = probs
            
            zero_count = (probs <= 1e-9).sum()
            if zero_count > 0:
                print(f"  [Feature: {col}]")
                print(f"    → {zero_count:,} values ({zero_count/len(df)*100:.1f}%) have near-zero clean probability")
                print(f"    → These are ANOMALOUS values not seen in clean traffic")
            
            df_scored['P_clean_joint'] *= probs

        zero_joint = (df_scored['P_clean_joint'] <= 0).sum()
        if zero_joint > 0:
            print(f"\n  [Joint Probability]")
            print(f"    → {zero_joint:,} rows ({zero_joint/len(df)*100:.1f}%) have zero joint clean probability")
            print(f"    → These are HIGHLY ANOMALOUS combinations")
            df_scored['P_clean_joint'] = df_scored['P_clean_joint'].replace(0, 1e-10)

        # Calculate observed frequencies
        print(f"\n[Calculating Observed Frequencies]")
        tuple_str_cols = []
        for col in tuple_cols:
            col_str = f'{col}_str'
            df_scored[col_str] = df_scored[col].astype(str).fillna('nan')
            tuple_str_cols.append(col_str)

        counts = df_scored.groupby(tuple_str_cols, sort=False).size()
        total_rows = float(len(df_scored))
        P_obs = (counts / total_rows).rename('P_obs_joint').reset_index()

        df_scored = df_scored.merge(P_obs, on=tuple_str_cols, how='left')
        df_scored['P_obs_joint'] = df_scored['P_obs_joint'].fillna(1.0 / total_rows)

        # Compute final bot odds
        print(f"\n[Computing Bot Likelihood Scores]")
        alpha = 0.5

        P_obs_arr = df_scored['P_obs_joint'].to_numpy(dtype=float)
        P_clean_arr = df_scored['P_clean_joint'].to_numpy(dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            scores = (P_obs_arr / (alpha * P_clean_arr)) - 1.0

        scores = np.clip(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
        df_scored['pros_anomaly_score'] = scores

        # Clean up temporary columns
        for col in tuple_str_cols:
            if col in df_scored.columns:
                df_scored.drop(columns=[col], inplace=True)

        df_scored.sort_values('pros_anomaly_score', ascending=False, inplace=True)

        print(f"\n{'='*70}")
        print(f"✓ SCORING COMPLETE:")
        print(f"{'='*70}")
        print(f"  Score Range: [{df_scored['pros_anomaly_score'].min():.4f}, {df_scored['pros_anomaly_score'].max():.4f}]")
        print(f"  Mean Score: {df_scored['pros_anomaly_score'].mean():.4f}")
        print(f"  Median Score: {df_scored['pros_anomaly_score'].median():.4f}")
        print(f"  90th Percentile: {df_scored['pros_anomaly_score'].quantile(0.9):.4f}")
        print(f"  95th Percentile: {df_scored['pros_anomaly_score'].quantile(0.95):.4f}")
        print(f"  99th Percentile: {df_scored['pros_anomaly_score'].quantile(0.99):.4f}")
        print(f"{'='*70}")

        return df_scored

    def generate_interpretable_rules(self, df_scored, top_n=20):
        """
        Generate human-interpretable rules for top anomalies.
        Directly addresses RQ3: Interpretability of PROS technique.
        """
        print("\n" + "="*70)
        print("GENERATING INTERPRETABLE RULES (RQ3)")
        print("="*70)
        print("\nObjective: Translate anomaly scores into actionable, human-readable rules")
        print("="*70 + "\n")

        required_cols = ['channelID', 'pros_anomaly_score', 'channelTitle', 'videoGenre', 
                        'is_duplicate', 'account_age_bin', 'comments_per_video_bin']
        available_cols = [c for c in required_cols if c in df_scored.columns]
        subset = df_scored[available_cols].copy()

        # Aggregate by channel
        agg_dict = {'pros_anomaly_score': 'max'}
        if 'channelTitle' in subset.columns:
            agg_dict['channelTitle'] = 'first'
        if 'videoGenre' in subset.columns:
            agg_dict['videoGenre'] = 'first'
        if 'is_duplicate' in subset.columns:
            agg_dict['is_duplicate'] = 'mean'
        if 'account_age_bin' in subset.columns:
            agg_dict['account_age_bin'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        if 'comments_per_video_bin' in subset.columns:
            agg_dict['comments_per_video_bin'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]

        channel_scores = subset.groupby('channelID', sort=False).agg(agg_dict).reset_index()
        channel_scores = channel_scores.sort_values('pros_anomaly_score', ascending=False)

        rules = []
        print("[Top Suspicious Channels]\n")
        
        for i, row in channel_scores.head(top_n).iterrows():
            rule_parts = []
            
            # Build interpretable rule
            if 'account_age_bin' in row:
                rule_parts.append(f"Account Age: {row['account_age_bin']}")
            if 'comments_per_video_bin' in row:
                rule_parts.append(f"Velocity: {row['comments_per_video_bin']} comments/video")
            if 'is_duplicate' in row and row['is_duplicate'] > 0:
                rule_parts.append(f"Duplicates: {row['is_duplicate']:.1%}")
            
            rule_text = " AND ".join(rule_parts) if rule_parts else "Multiple anomaly indicators"
            
            rule = {
                'channel': str(row.get('channelTitle', ''))[:50],
                'genre': str(row.get('videoGenre', '')),
                'score': float(row['pros_anomaly_score']),
                'rule': rule_text,
                'account_age': str(row.get('account_age_bin', 'Unknown')),
                'velocity': str(row.get('comments_per_video_bin', 'Unknown'))
            }
            rules.append(rule)
            
            # Print human-readable rule
            if i < 10:  # Print top 10
                print(f"  #{i+1}. Channel: {rule['channel']}")
                print(f"       Score: {rule['score']:.3f}")
                print(f"       Rule: {rule['rule']}")
                print(f"       Genre: {rule['genre']}\n")
        
        # Store RQ3 metrics
        self.rq_metrics['rq3_interpretability'] = {
            'total_rules_generated': len(rules),
            'avg_score_top_10': np.mean([r['score'] for r in rules[:10]]) if rules else 0,
            'features_in_rules': list(agg_dict.keys())
        }
        
        print(f"{'='*70}")
        print(f"✓ Generated {len(rules)} interpretable rules")
        print(f"{'='*70}")
        
        return rules

    def analyze_channel_categories(self, df_scored):
        """
        Analyze bot likelihood across different YouTube categories/genres.
        Directly addresses RQ4: Which categories have the most bot traffic?
        """
        print("\n" + "="*70)
        print("ANALYZING CHANNEL CATEGORIES (RQ4)")
        print("="*70)
        print("\nObjective: Identify which YouTube genres/categories have highest bot activity")
        print("="*70 + "\n")

        genre_col = 'genre_group' if 'genre_group' in df_scored.columns else 'videoGenre'
        if genre_col not in df_scored.columns:
            print("⚠ Warning: No genre information available")
            return None

        # Aggregate by genre
        grouped = df_scored.groupby(genre_col)
        genre_stats = grouped.agg(
            avg_bot_score=('pros_anomaly_score', 'mean'),
            std_bot_score=('pros_anomaly_score', 'std'),
            total_comments=('pros_anomaly_score', 'count'),
            duplicate_rate=('is_duplicate', 'mean'),
            unique_channels=('channelID', 'nunique'),
        ).round(3).reset_index()

        genre_stats = genre_stats.sort_values('avg_bot_score', ascending=False)

        print("[Genre-Level Bot Analysis]\n")
        
        for i, row in genre_stats.head(10).iterrows():
            genre = row[genre_col]
            score = row['avg_bot_score']
            comments = row['total_comments']
            channels = row['unique_channels']
            dup_rate = row['duplicate_rate']
            
            # Interpret risk level
            if score > 1.0:
                risk = "🔴 VERY HIGH"
            elif score > 0.5:
                risk = "🟠 HIGH"
            elif score > 0.2:
                risk = "🟡 MEDIUM"
            else:
                risk = "🟢 LOW"
            
            print(f"  {i+1}. {genre}")
            print(f"     Risk Level: {risk} (Score: {score:.3f})")
            print(f"     Volume: {comments:,} comments from {channels:,} channels")
            print(f"     Duplicate Rate: {dup_rate:.1%}")
            print()
        
        # Store RQ4 metrics
        if len(genre_stats) > 0:
            self.rq_metrics['rq4_genre_analysis'] = {
                'total_genres': len(genre_stats),
                'highest_risk_genre': genre_stats.iloc[0][genre_col],
                'highest_risk_score': float(genre_stats.iloc[0]['avg_bot_score']),
                'lowest_risk_genre': genre_stats.iloc[-1][genre_col],
                'lowest_risk_score': float(genre_stats.iloc[-1]['avg_bot_score'])
            }

        print(f"{'='*70}")
        print(f"✓ Analyzed {len(genre_stats)} genres/categories")
        print(f"{'='*70}")

        return genre_stats

    def compare_with_isolation_forest(self, df):
        """Compare PROS with Isolation Forest baseline."""
        try:
            from sklearn.ensemble import IsolationForest

            print("\n" + "="*70)
            print("BASELINE COMPARISON: Isolation Forest")
            print("="*70)

            categorical_features = [
                'account_age_bin', 'subscriber_bin', 'video_count_bin',
                'time_of_day', 'profile_complete_bin'
            ]
            available_features = [f for f in categorical_features if f in df.columns]

            if not available_features:
                print("⚠ No features available for Isolation Forest")
                return df

            features_df = df[available_features].copy()

            for col in available_features:
                col_data = features_df[col]
                if pd.api.types.is_categorical_dtype(col_data):
                    features_df[col] = col_data.cat.add_categories(['Missing']).fillna('Missing')
                else:
                    features_df[col] = col_data.astype(str).fillna('Missing')

            features_df = pd.get_dummies(features_df, columns=available_features, drop_first=True)

            if features_df.shape[1] == 0:
                print("⚠ No features after encoding")
                return df

            if len(features_df) < 10:
                print(f"⚠ Not enough data ({len(features_df)} samples)")
                return df

            contamination = 0.1
            print(f"\n[Configuration]")
            print(f"  Features: {available_features}")
            print(f"  Contamination rate: {contamination:.1%}")
            print(f"  Encoded dimensions: {features_df.shape[1]}")

            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                n_jobs=-1
            )
            
            print(f"\n[Training Isolation Forest...]")
            iso_scores = iso_forest.fit_predict(features_df)

            df['iso_forest_score'] = iso_scores
            df['iso_forest_anomaly'] = (iso_scores == -1).astype(np.int8)

            if 'pros_anomaly_score' in df.columns:
                pros_threshold = df['pros_anomaly_score'].quantile(0.9)
                df['pros_anomaly_binary'] = (df['pros_anomaly_score'] > pros_threshold).astype(np.int8)

                agreement = (df['iso_forest_anomaly'] == df['pros_anomaly_binary']).mean()
                
                pros_flags = df['pros_anomaly_binary'].sum()
                iso_flags = df['iso_forest_anomaly'].sum()
                total = len(df)
                
                print(f"\n[Comparison Results]")
                print(f"  Agreement rate: {agreement:.1%}")
                print(f"  PROS flagged: {pros_flags:,} ({pros_flags / total:.1%})")
                print(f"  Isolation Forest flagged: {iso_flags:,} ({iso_flags / total:.1%})")

                true_pos = ((df['pros_anomaly_binary'] == 1) & (df['iso_forest_anomaly'] == 1)).sum()
                false_pos = ((df['pros_anomaly_binary'] == 0) & (df['iso_forest_anomaly'] == 1)).sum()
                false_neg = ((df['pros_anomaly_binary'] == 1) & (df['iso_forest_anomaly'] == 0)).sum()

                if (true_pos + false_pos) > 0:
                    precision = true_pos / (true_pos + false_pos)
                    print(f"  Precision (IF vs PROS): {precision:.1%}")

                if (true_pos + false_neg) > 0:
                    recall = true_pos / (true_pos + false_neg)
                    print(f"  Recall (IF vs PROS): {recall:.1%}")

            print(f"{'='*70}")
            return df

        except Exception as e:
            print(f"⚠ Error running Isolation Forest: {e}")
            return df

    def generate_research_summary(self):
        """
        Generate a comprehensive summary linking results to research questions.
        """
        print("\n" + "="*70)
        print("RESEARCH QUESTIONS SUMMARY")
        print("="*70)
        
        print("\n[RQ1: Unsupervised Anomaly Detection]")
        print("  Question: How can unsupervised anomaly detection identify automated")
        print("           commenting behaviour without relying on labelled data?")
        print("\n  Answer:")
        rq1 = self.rq_metrics.get('rq1_unsupervised', {})
        if rq1:
            print(f"    ✓ Successfully identified clean distributions for {rq1.get('successful_features', 0)} features")
            print(f"    ✓ Success rate: {rq1.get('success_rate', 0):.1%}")
            print(f"    ✓ Method: Algorithm 1 (find unattacked bins) + Algorithm 2 (anomaly scoring)")
            print(f"    ✓ Features used: {', '.join(rq1.get('features_used', []))}")
        
        print("\n[RQ2: Structural and Behavioral Features]")
        print("  Question: Which features most effectively differentiate bots from humans?")
        print("\n  Answer:")
        rq2 = self.rq_metrics.get('rq2_feature_importance', {})
        if rq2:
            # Sort by effectiveness
            sorted_features = sorted(rq2.items(), key=lambda x: x[1]['effectiveness_score'], reverse=True)
            print("    Feature Importance (by unattacked bin discovery):")
            for feature, metrics in sorted_features[:5]:
                score = metrics['effectiveness_score']
                bins = metrics['unattacked_bins']
                effectiveness = "High" if score > 0.3 else "Medium" if score > 0.15 else "Low"
                print(f"      • {feature}: {effectiveness} ({bins} clean bins, {score:.1%} effectiveness)")
        
        print("\n[RQ3: Interpretability]")
        print("  Question: How can PROS remain interpretable to human analysts?")
        print("\n  Answer:")
        rq3 = self.rq_metrics.get('rq3_interpretability', {})
        if rq3:
            print(f"    ✓ Generated {rq3.get('total_rules_generated', 0)} human-readable rules")
            print(f"    ✓ Average anomaly score (top 10): {rq3.get('avg_score_top_10', 0):.3f}")
            print(f"    ✓ Rules based on: {', '.join(rq3.get('features_in_rules', []))}")
            print(f"    ✓ Format: IF [Feature Combination] THEN [Bot Likelihood = X]")
        
        print("\n[RQ4: Category Analysis]")
        print("  Question: What YouTube channel categories have the most bot traffic?")
        print("\n  Answer:")
        rq4 = self.rq_metrics.get('rq4_genre_analysis', {})
        if rq4:
            print(f"    ✓ Analyzed {rq4.get('total_genres', 0)} categories")
            print(f"    ✓ Highest risk: {rq4.get('highest_risk_genre', 'N/A')} (score: {rq4.get('highest_risk_score', 0):.3f})")
            print(f"    ✓ Lowest risk: {rq4.get('lowest_risk_genre', 'N/A')} (score: {rq4.get('lowest_risk_score', 0):.3f})")
        
        print("\n" + "="*70)

    def run_full_analysis(self, df):
        """Run complete PROS analysis pipeline with RQ tracking."""
        print("="*70)
        print("PROS BOT DETECTION ANALYSIS")
        print("Research Questions from Mid-Term Report")
        print("="*70)

        df_engineered = self.engineer_features(df)
        self.define_conditional_independence(df_engineered)
        self.estimate_clean_distributions(df_engineered)
        df_scored = self.calculate_anomaly_scores(df_engineered)
        rules = self.generate_interpretable_rules(df_scored)
        genre_analysis = self.analyze_channel_categories(df_scored)

        # Baseline comparison
        sample_size = min(len(df_scored), 100000)
        df_sample = df_scored.sample(sample_size, random_state=42) if len(df_scored) > sample_size else df_scored
        df_comparison = self.compare_with_isolation_forest(df_sample)

        # Generate RQ summary
        self.generate_research_summary()

        return {
            'df_scored': df_scored,
            'rules': rules,
            'genre_analysis': genre_analysis,
            'clean_distributions': self.clean_distributions,
            'unattacked_bins': self.unattacked_bins,
            'feature_stats': self.feature_stats,
            'df_comparison': df_comparison,
            'rq_metrics': self.rq_metrics
        }



def visualize_results(results):
    """Create two clear visualizations to avoid overlapping text."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import gridspec

        df_scored = results['df_scored']
        genre_analysis = results['genre_analysis']
        rules = results['rules']
        feature_stats = results.get('feature_stats', {})
        rq_metrics = results.get('rq_metrics', {})

        # Modern color palette
        colors = {
            'primary': '#2C3E50',
            'secondary': '#E74C3C',
            'accent': '#3498DB',
            'success': '#27AE60',
            'warning': '#F39C12',
            'background': '#ECF0F1',
            'text': '#2C3E50'
        }

        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # ====================================================
        # VISUALIZATION 1: RESEARCH QUESTIONS OVERVIEW
        # ====================================================
        print("\n" + "="*70)
        print("CREATING VISUALIZATION 1: RESEARCH QUESTIONS OVERVIEW")
        print("="*70)
        
        fig1 = plt.figure(figsize=(18, 10))
        gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.25, wspace=0.25)

        # --- PLOT 1: Score Distribution (RQ1) ---
        ax1 = fig1.add_subplot(gs1[0, :])
        
        if 'pros_anomaly_score' in df_scored.columns:
            scores = df_scored['pros_anomaly_score']
            
            # Create histogram
            n, bins, patches = ax1.hist(scores, bins=80, alpha=0.7, color=colors['accent'], 
                                       edgecolor='black', linewidth=0.5, density=True)
            
            # Color bins by risk level
            for i, patch in enumerate(patches):
                if bins[i] > scores.quantile(0.95):
                    patch.set_facecolor(colors['secondary'])
                elif bins[i] > scores.quantile(0.90):
                    patch.set_facecolor(colors['warning'])
            
            # Add threshold lines
            for q, label, color in [(0.90, 'Top 10%', colors['warning']), 
                                    (0.95, 'Top 5%', colors['secondary'])]:
                threshold = scores.quantile(q)
                ax1.axvline(x=threshold, color=color, linestyle='--', linewidth=2.5, 
                           label=f'{label}: {threshold:.3f}')
            
            ax1.set_xlabel('PROS Anomaly Score', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax1.set_title('RQ1: Unsupervised Bot Detection - Score Distribution', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.legend(fontsize=10, framealpha=0.9, loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics box (positioned to avoid overlap)
            stats_text = f'Total Samples: {len(df_scored):,}\nMean: {scores.mean():.3f}\nMedian: {scores.median():.3f}\n90th %ile: {scores.quantile(0.9):.3f}\nStd Dev: {scores.std():.3f}'
            ax1.text(0.98, 0.85, stats_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                            edgecolor=colors['primary'], linewidth=1.5))

        # --- PLOT 2: Feature Importance (RQ2) ---
        ax2 = fig1.add_subplot(gs1[1, 0])
        
        if feature_stats:
            features = list(feature_stats.keys())
            importance = [stats['num_unattacked_bins'] for stats in feature_stats.values()]
            
            # Sort by importance
            sorted_pairs = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
            features_sorted, importance_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            # Limit to top 8 features for clarity
            if len(features_sorted) > 8:
                features_sorted = features_sorted[:8]
                importance_sorted = importance_sorted[:8]
            
            # Color by effectiveness
            bar_colors = [colors['success'] if val > 5 else colors['warning'] if val > 2 else colors['secondary'] 
                         for val in importance_sorted]
            
            bars = ax2.barh(range(len(features_sorted)), importance_sorted, 
                          color=bar_colors, edgecolor='black', linewidth=0.5, height=0.6)
            ax2.set_yticks(range(len(features_sorted)))
            
            # Format feature names for better readability
            feature_labels = []
            for f in features_sorted:
                label = f.replace('_', ' ').title()
                if len(label) > 15:
                    label = label[:12] + '...'
                feature_labels.append(label)
            
            ax2.set_yticklabels(feature_labels, fontsize=9)
            ax2.set_xlabel('Unattacked Bins Found', fontsize=10, fontweight='bold')
            ax2.set_title('RQ2: Feature Effectiveness\n(# of Clean Bins)', 
                         fontsize=12, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, importance_sorted)):
                ax2.text(val, i, f' {val}', va='center', fontsize=9, fontweight='bold')

        # --- PLOT 3: Genre Risk Analysis (RQ4) ---
        ax3 = fig1.add_subplot(gs1[1, 1])
        
        if genre_analysis is not None and len(genre_analysis) > 0:
            top_genres = genre_analysis.head(8)  # Limit to 8 for clarity
            genre_col = genre_analysis.columns[0]
            
            x_pos = range(len(top_genres))
            scores = top_genres['avg_bot_score'].values
            
            # Color by risk level
            bar_colors = []
            for score in scores:
                if score > 1.0:
                    bar_colors.append(colors['secondary'])
                elif score > 0.5:
                    bar_colors.append(colors['warning'])
                elif score > 0.2:
                    bar_colors.append('#FFD700')  # Gold for medium
                else:
                    bar_colors.append(colors['success'])
            
            bars = ax3.barh(x_pos, scores, color=bar_colors, edgecolor='black', linewidth=1, height=0.6)
            
            # Format genre names for readability
            genre_labels = []
            for genre in top_genres[genre_col]:
                label = str(genre)
                if len(label) > 15:
                    label = label[:12] + '...'
                genre_labels.append(label)
            
            ax3.set_yticks(x_pos)
            ax3.set_yticklabels(genre_labels, fontsize=9)
            ax3.set_xlabel('Avg Bot Score', fontsize=10, fontweight='bold')
            ax3.set_title('RQ4: Top Risky Genres', 
                         fontsize=12, fontweight='bold', pad=15)
            ax3.grid(True, alpha=0.3, axis='x')
            ax3.invert_yaxis()
            
            # Add risk threshold lines
            ax3.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
            ax3.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                risk = "HIGH" if score > 1.0 else "MED" if score > 0.5 else "LOW"
                ax3.text(score, i, f' {score:.2f}', va='center', fontsize=8, fontweight='bold')

        # --- PLOT 4: RQ Summary Dashboard ---
        ax4 = fig1.add_subplot(gs1[1, 2])
        ax4.axis('off')
        
        # Create comprehensive summary text
        summary_text = "RESEARCH QUESTIONS SUMMARY\n" + "="*40 + "\n\n"
        
        # RQ1 Summary
        rq1 = rq_metrics.get('rq1_unsupervised', {})
        if rq1:
            summary_text += "RQ1: Unsupervised Detection\n"
            summary_text += f"  • Success: {rq1.get('successful_features', 0)}/{rq1.get('total_features', 0)} features\n"
            summary_text += f"  • Rate: {rq1.get('success_rate', 0)*100:.0f}%\n"
            summary_text += f"  • Method: PROS Algo 1 + 2\n\n"
        
        
        # RQ3 Summary
        rq3 = rq_metrics.get('rq3_interpretability', {})
        if rq3:
            summary_text += "RQ3: Interpretability\n"
            summary_text += f"  • Rules: {rq3.get('total_rules_generated', 0)}\n"
            summary_text += f"  • Avg Score: {rq3.get('avg_score_top_10', 0):.3f}\n\n"
        
        # RQ4 Summary
        rq4 = rq_metrics.get('rq4_genre_analysis', {})
        if rq4:
            summary_text += "RQ4: Genre Analysis\n"
            summary_text += f"  • Genres: {rq4.get('total_genres', 0)}\n"
            summary_text += f"  • Highest Risk: {rq4.get('highest_risk_genre', 'N/A')}\n"
            summary_text += f"  • Risk Score: {rq4.get('highest_risk_score', 0):.3f}"
        
        # Add detection statistics if available
        if 'pros_anomaly_score' in df_scored.columns:
            high_risk = (df_scored['pros_anomaly_score'] > 0.5).sum()
            total = len(df_scored)
            summary_text += f"\n\nDETECTION SUMMARY\n" + "="*40 + "\n"
            summary_text += f"  • High Risk (>0.5): {high_risk:,} ({high_risk/total*100:.1f}%)\n"
            summary_text += f"  • Total Samples: {total:,}"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace', linespacing=1.5,
                bbox=dict(boxstyle='round', facecolor=colors['background'], 
                         alpha=0.9, edgecolor=colors['primary'], linewidth=2))

        # Add main title
        fig1.suptitle('PROS Bot Detection - Research Questions Overview', 
                     fontsize=18, fontweight='bold', y=0.98)

        # Save first visualization
        viz1_path = os.path.join(YOUTUBE_RESULTS_DIR, "pros_visualization_1_rq_overview.png")
        plt.savefig(viz1_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        print(f"✓ Visualization 1 saved: {viz1_path}")

        # ====================================================
        # VISUALIZATION 2: DETAILED FEATURE ANALYSIS
        # ====================================================
        print("\n" + "="*70)
        print("CREATING VISUALIZATION 2: DETAILED FEATURE ANALYSIS")
        print("="*70)
        
        fig2 = plt.figure(figsize=(16, 10))
        gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.25)

        # --- PLOT 1: Account Age Analysis ---
        ax1 = fig2.add_subplot(gs2[0, 0])
        
        if 'account_age_bin' in df_scored.columns and 'pros_anomaly_score' in df_scored.columns:
            # Create age groups with consistent ordering
            age_order = ['<1d', '1-7d', '1w-1m', '1-3m', '3m-1y', '>1y']
            existing_ages = [age for age in age_order if age in df_scored['account_age_bin'].unique()]
            
            if existing_ages:
                age_data = df_scored.groupby('account_age_bin')['pros_anomaly_score'].agg(
                    ['mean', 'std', 'count']
                ).reindex(existing_ages)
                
                # Create bar plot
                bars = ax1.bar(range(len(existing_ages)), age_data['mean'], 
                             color=colors['accent'], edgecolor='black', linewidth=0.8, width=0.7)
                
                # Color bars by risk
                for i, (bar, score) in enumerate(zip(bars, age_data['mean'])):
                    if score > 0.5:
                        bar.set_color(colors['secondary'])
                    elif score > 0.2:
                        bar.set_color(colors['warning'])
                
                # Add error bars (standard deviation)
                ax1.errorbar(range(len(existing_ages)), age_data['mean'], 
                           yerr=age_data['std'], fmt='none', ecolor='black', 
                           capsize=5, capthick=1.5, elinewidth=1)
                
                # Set labels and formatting
                ax1.set_xticks(range(len(existing_ages)))
                ax1.set_xticklabels(existing_ages, rotation=45, ha='right', fontsize=9)
                ax1.set_ylabel('Average Bot Score', fontsize=11, fontweight='bold')
                ax1.set_xlabel('Account Age', fontsize=11, fontweight='bold')
                ax1.set_title('Account Age vs Bot Likelihood', 
                            fontsize=13, fontweight='bold', pad=15)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add count labels above bars
                for i, (bar, count) in enumerate(zip(bars, age_data['count'])):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'n={count:,}', ha='center', va='bottom', fontsize=8, 
                            rotation=0)
        
        # --- PLOT 2: Posting Velocity Analysis ---
        ax2 = fig2.add_subplot(gs2[0, 1])
        
        if 'comments_per_video_bin' in df_scored.columns:
            # Define consistent ordering for velocity bins
            velocity_order = ['1', '2-3', '4-5', '6-10', '>10']
            existing_vel = [vel for vel in velocity_order if vel in df_scored['comments_per_video_bin'].unique()]
            
            if existing_vel:
                velocity_data = df_scored.groupby('comments_per_video_bin')['pros_anomaly_score'].agg(
                    ['mean', 'std', 'count']
                ).reindex(existing_vel)
                
                # Create bar plot
                bars = ax2.bar(range(len(existing_vel)), velocity_data['mean'], 
                             color=colors['accent'], edgecolor='black', linewidth=0.8, width=0.7)
                
                # Color bars by risk
                for i, (bar, score) in enumerate(zip(bars, velocity_data['mean'])):
                    if score > 0.5:
                        bar.set_color(colors['secondary'])
                
                # Add error bars
                ax2.errorbar(range(len(existing_vel)), velocity_data['mean'], 
                           yerr=velocity_data['std'], fmt='none', ecolor='black', 
                           capsize=5, capthick=1.5, elinewidth=1)
                
                # Set labels and formatting
                ax2.set_xticks(range(len(existing_vel)))
                ax2.set_xticklabels(existing_vel, rotation=45, ha='right', fontsize=9)
                ax2.set_ylabel('Average Bot Score', fontsize=11, fontweight='bold')
                ax2.set_xlabel('Comments per Video', fontsize=11, fontweight='bold')
                ax2.set_title('Posting Velocity vs Bot Likelihood', 
                            fontsize=13, fontweight='bold', pad=15)
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, velocity_data['count'])):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'n={count:,}', ha='center', va='bottom', fontsize=8)
        
        # --- PLOT 3: Profile Completeness Analysis ---
        ax3 = fig2.add_subplot(gs2[1, 0])
        
        if 'profile_complete_bin' in df_scored.columns:
            # Define consistent ordering for profile completeness
            profile_order = ['None', 'Low', 'Medium', 'High']
            existing_profiles = [prof for prof in profile_order if prof in df_scored['profile_complete_bin'].unique()]
            
            if existing_profiles:
                profile_data = df_scored.groupby('profile_complete_bin')['pros_anomaly_score'].agg(
                    ['mean', 'std', 'count']
                ).reindex(existing_profiles)
                
                # Create bar plot
                bars = ax3.bar(range(len(existing_profiles)), profile_data['mean'], 
                             color=colors['accent'], edgecolor='black', linewidth=0.8, width=0.7)
                
                # Color bars by risk
                for i, (bar, score) in enumerate(zip(bars, profile_data['mean'])):
                    if score > 0.5:
                        bar.set_color(colors['secondary'])
                
                # Add error bars
                ax3.errorbar(range(len(existing_profiles)), profile_data['mean'], 
                           yerr=profile_data['std'], fmt='none', ecolor='black', 
                           capsize=5, capthick=1.5, elinewidth=1)
                
                # Set labels and formatting
                ax3.set_xticks(range(len(existing_profiles)))
                ax3.set_xticklabels(existing_profiles, fontsize=10)
                ax3.set_ylabel('Average Bot Score', fontsize=11, fontweight='bold')
                ax3.set_xlabel('Profile Completeness', fontsize=11, fontweight='bold')
                ax3.set_title('Profile Completeness vs Bot Likelihood', 
                            fontsize=13, fontweight='bold', pad=15)
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, profile_data['count'])):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'n={count:,}', ha='center', va='bottom', fontsize=8)
        
        # --- PLOT 4: Top Channel Rules ---
        ax4 = fig2.add_subplot(gs2[1, 1])
        ax4.axis('off')
        
        if rules and len(rules) > 0:
            # Prepare rules text
            rules_text = "TOP SUSPICIOUS CHANNELS\n" + "="*40 + "\n\n"
            
            for i, rule in enumerate(rules[:6]):  # Show top 6 rules
                rules_text += f"[{i+1}] {rule['channel']}\n"
                rules_text += f"    Score: {rule['score']:.3f}\n"
                
                # Truncate rule if too long
                rule_display = rule['rule']
                if len(rule_display) > 40:
                    rule_display = rule_display[:37] + "..."
                
                rules_text += f"    Rule: {rule_display}\n"
                
                # Add genre if available
                if rule['genre'] and rule['genre'] != 'Unknown':
                    rules_text += f"    Genre: {rule['genre'][:20]}\n"
                
                rules_text += "\n"
            
            # Add additional stats
            rules_text += "="*40 + "\n"
            if len(rules) > 6:
                rules_text += f"+ {len(rules) - 6} more channels...\n\n"
            
            # Calculate average scores
            top_scores = [r['score'] for r in rules[:10]]
            if top_scores:
                rules_text += f"Avg Top 10 Score: {np.mean(top_scores):.3f}\n"
                rules_text += f"Max Score: {max(top_scores):.3f}"
            
            ax4.text(0.05, 0.95, rules_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace', linespacing=1.4,
                    bbox=dict(boxstyle='round', facecolor=colors['background'], 
                             alpha=0.9, edgecolor=colors['primary'], linewidth=2))
        else:
            ax4.text(0.5, 0.5, "No interpretable rules generated", 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor=colors['background'], alpha=0.8))

        # Add main title
        fig2.suptitle('PROS Bot Detection - Detailed Feature Analysis', 
                     fontsize=18, fontweight='bold', y=0.98)

        # Save second visualization
        viz2_path = os.path.join(YOUTUBE_RESULTS_DIR, "pros_visualization_2_feature_analysis.png")
        plt.savefig(viz2_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"✓ Visualization 2 saved: {viz2_path}")

        # ====================================================
        # PRINT VISUALIZATION SUMMARY
        # ====================================================
        print("\n" + "="*70)
        print("VISUALIZATION SUMMARY")
        print("="*70)
        print("\nCreated 2 clear, non-overlapping visualizations:")
        print("\n[Visualization 1: Research Questions Overview]")
        print("  • RQ1: Score distribution with statistics")
        print("  • RQ2: Feature effectiveness (unattacked bins)")
        print("  • RQ4: Top risky genres")
        print("  • RQ Summary: Key metrics dashboard")
        print(f"  → Saved as: {viz1_path}")
        
        print("\n[Visualization 2: Detailed Feature Analysis]")
        print("  • Account Age: Bot likelihood by account age")
        print("  • Posting Velocity: Comments per video analysis")
        print("  • Profile Completeness: Profile quality impact")
        print("  • Top Channels: Interpretable detection rules")
        print(f"  → Saved as: {viz2_path}")
        print("\n" + "="*70)

        # Display both visualizations
        try:
            # Show first visualization
            img1 = plt.imread(viz1_path)
            fig_display1, ax_display1 = plt.subplots(figsize=(15, 8))
            ax_display1.imshow(img1)
            ax_display1.axis('off')
            plt.title('Visualization 1: Research Questions Overview', fontsize=14, fontweight='bold')
            plt.show()
            
            # Show second visualization
            img2 = plt.imread(viz2_path)
            fig_display2, ax_display2 = plt.subplots(figsize=(13, 8))
            ax_display2.imshow(img2)
            ax_display2.axis('off')
            plt.title('Visualization 2: Detailed Feature Analysis', fontsize=14, fontweight='bold')
            plt.show()
            
        except Exception as e:
            print(f"Note: Could not display images interactively. Files saved at:")
            print(f"  • {viz1_path}")
            print(f"  • {viz2_path}")

    except ImportError as e:
        print(f"⚠ Visualization libraries not available: {e}")
    except Exception as e:
        print(f"⚠ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TERMINAL_OUTPUT_PATH = os.path.join(
        YOUTUBE_RESULTS_DIR,
        f"pros_terminal_output_{timestamp}.txt"
    )

    class Logger(object):
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    logger = Logger(TERMINAL_OUTPUT_PATH)
    sys.stdout = logger

    try:
        if not os.path.exists(CSV_PATH):
            print(f" Error: CSV file not found at {CSV_PATH}")
            print("Please run the data fetching script first.")
        else:
            print(f"\n{'='*70}")
            print("LOADING DATA")
            print(f"{'='*70}")
            print(f"Source: {CSV_PATH}")
            
            df = pd.read_csv(CSV_PATH)

            if df.empty:
                print(" CSV file is empty.")
            else:
                print(f"✓ Loaded {len(df):,} rows with {len(df.columns)} columns")
                print(f"✓ Unique channels: {df['channelID'].nunique() if 'channelID' in df.columns else 'N/A':,}")
                print(f"{'='*70}\n")

                detector = PROSDetector(min_samples_per_bin=5, jsd_threshold=0.15)
                results = detector.run_full_analysis(df)

                # Save results
                if results.get('genre_analysis') is not None:
                    genre_csv_path = os.path.join(YOUTUBE_RESULTS_DIR, f"pros_genre_analysis_{timestamp}.csv")
                    results['genre_analysis'].to_csv(genre_csv_path, index=False)
                    print(f"\n✓ Saved genre analysis: {genre_csv_path}")

                if 'clean_distributions' in results:
                    clean_dist_path = os.path.join(YOUTUBE_RESULTS_DIR, f"pros_clean_distributions_{timestamp}.json")
                    with open(clean_dist_path, 'w') as f:
                        json.dump(results['clean_distributions'], f, indent=2)
                    print(f"✓ Saved clean distributions: {clean_dist_path}")
                
                if 'rq_metrics' in results:
                    rq_metrics_path = os.path.join(YOUTUBE_RESULTS_DIR, f"pros_rq_metrics_{timestamp}.json")
                    with open(rq_metrics_path, 'w') as f:
                        json.dump(results['rq_metrics'], f, indent=2)
                    print(f"✓ Saved RQ metrics: {rq_metrics_path}")

                visualize_results(results)
                
                print(f"\n{'='*70}")
                print("ANALYSIS COMPLETE!")
                print(f"{'='*70}")
                
    except Exception as e:
        print(f"\n⚠ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"\n✓ All results saved to: {YOUTUBE_RESULTS_DIR}")
        print(f"✓ Terminal output saved to: {TERMINAL_OUTPUT_PATH}")