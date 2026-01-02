import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from Youtube_model import PROSDetector
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)   
YOUTUBE_RESULTS_DIR = os.path.join(PARENT_DIR, "Youtube Results")
os.makedirs(YOUTUBE_RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
BOT_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
TOP_K_THRESHOLDS = [0.05, 0.10, 0.15]
N_CHANNELS = 1000
COMMENTS_PER_CHANNEL = 5000

# ============================================================
# IMPROVED SYNTHETIC DATA GENERATION
# ============================================================
def generate_realistic_synthetic_youtube_data(n_channels=N_CHANNELS,
                                              comments_per_channel=COMMENTS_PER_CHANNEL,
                                              bot_fraction=0.1,
                                              random_seed=RANDOM_SEED):
    rng = np.random.default_rng(random_seed + int(bot_fraction*1000))
    total_comments = n_channels * comments_per_channel

    # Channel-level data with realistic distributions
    channel_data = pd.DataFrame({
        'channelID': [f'CH_{i:04d}' for i in range(n_channels)],
        'is_bot': rng.random(n_channels) < bot_fraction,
        'channelDate': pd.to_datetime('2020-01-01') + pd.to_timedelta(rng.integers(0, 1200, n_channels), 'D'),
        'channelViewCount': rng.exponential(10000, n_channels).astype(np.int32),
        'channelSubscriberCount': rng.exponential(1000, n_channels).astype(np.int32),
        'channelVideoCount': rng.exponential(50, n_channels).astype(np.int32),
        'channelCountry': rng.choice(['US','UK','CA','AU','DE','FR', None], n_channels)
    })

    # Comment-level data
    channel_idx = np.repeat(np.arange(n_channels), comments_per_channel)
    channel_series = channel_data.iloc[channel_idx]
    bot_mask = np.repeat(channel_data['is_bot'].values, comments_per_channel)
    
    # IMPROVEMENT 1: More realistic like counts
    # Bots: rarely get likes (0-2), Humans: often get likes (0-5)
    like_counts = np.where(
        bot_mask,
        rng.poisson(0.5, total_comments),  # Bots: avg 0.5 likes
        rng.poisson(2, total_comments)     # Humans: avg 2 likes
    )
    
    # IMPROVEMENT 2: Categorical duplicate behavior (not binary perfect)
    # Create 3 categories: 0=no duplicate, 1=minor duplicate, 2=major duplicate
    duplicate_pattern = np.zeros(total_comments, dtype=int)
    
    # Humans: mostly no duplicates, some minor
    human_mask = ~bot_mask
    duplicate_pattern[human_mask] = np.where(
        rng.random(np.sum(human_mask)) < 0.05,  # 5% humans have minor duplicates
        1,  # minor duplicate
        0   # no duplicate
    )
    
    # Bots: mix of no, minor, and major duplicates
    bot_duplicate_probs = rng.random(np.sum(bot_mask))
    duplicate_pattern[bot_mask] = np.select(
        [
            bot_duplicate_probs < 0.3,   # 30%: no duplicate
            bot_duplicate_probs < 0.7,   # 40%: minor duplicate
            bot_duplicate_probs >= 0.7   # 30%: major duplicate
        ],
        [0, 1, 2],
        default=0
    )
    
    # IMPROVEMENT 3: Add comment length feature (categorical)
    comment_length = np.where(
        bot_mask,
        rng.choice([1, 2, 3], total_comments, p=[0.6, 0.3, 0.1]),  # Bots: short comments
        rng.choice([1, 2, 3, 4], total_comments, p=[0.2, 0.3, 0.3, 0.2])  # Humans: varied
    )
    
    df = pd.DataFrame({
        'videoID': [f'VID_{c}_{k}' for c in range(n_channels) for k in range(comments_per_channel)],
        'channelID': channel_series['channelID'].values,
        'channelTitle': [f'Channel_{c}' for c in channel_idx],
        'channelDate': channel_series['channelDate'].values,
        'channelViewCount': channel_series['channelViewCount'].values,
        'channelSubscriberCount': channel_series['channelSubscriberCount'].values,
        'channelVideoCount': channel_series['channelVideoCount'].values,
        'channelCountry': channel_series['channelCountry'].values,
        'commentDate': channel_series['channelDate'].values + pd.to_timedelta(rng.integers(0, 1000, total_comments), 'D'),
        'commentLikeCount': like_counts,
        'videoGenre': rng.choice(['Entertainment','Gaming','Music','News & Politics','Howto & Style','Sports','Education'], total_comments),
        'hasDescription': rng.random(total_comments) < 0.8,
        'channelDescription': 'Some description',
        'defaultProfilePic': rng.random(total_comments) < 0.3,
        'commentText': 'Sample text',  # Not used by PROS
        'isDuplicateComment': duplicate_pattern,  # Now categorical (0,1,2)
        'commentLength': comment_length,  # New feature
        'label': bot_mask.astype(np.int8)
    })
    
    return df

# ============================================================
# CUSTOM PROSDETECTOR FOR SYNTHETIC DATA
# ============================================================
class SyntheticPROSDetector(PROSDetector):
    """Custom PROS detector optimized for synthetic validation"""
    
    def define_conditional_independence(self, df):
        """Override with synthetic-optimized rules"""
        print("\nUsing synthetic-optimized conditional independence rules...")
        
        # Use only categorical features (no binary, no perfect correlations)
        self.conditional_independence_rules = {
            'account_age_bin': ['time_of_day', 'subscriber_bin'],
            'subscriber_bin': ['genre_group', 'comments_per_video_bin'],
            'profile_complete_bin': ['account_age_bin', 'likes_bin'],
            'time_of_day': ['subscriber_bin', 'video_count_bin'],
            'comments_per_video_bin': ['genre_group', 'subscriber_bin'],
            'genre_group': ['account_age_bin', 'subscriber_bin']
        }
        
        # Filter to only available features
        available_features = set(df.columns)
        self.conditional_independence_rules = {
            k: [f for f in v if f in available_features]
            for k, v in self.conditional_independence_rules.items()
            if k in available_features and v
        }
        
        print("Optimized rules:")
        for feature, independents in self.conditional_independence_rules.items():
            if independents:  # Only show if there are independent features
                print(f"  {feature} ⊥ {independents}")
        
        return self.conditional_independence_rules
    
    def engineer_features(self, df):
        """Add synthetic-specific feature engineering"""
        df_engineered = super().engineer_features(df)
        
        # Add comment length bin from synthetic data
        if 'commentLength' in df.columns:
            df_engineered['comment_length_bin'] = pd.cut(
                df['commentLength'].fillna(1),
                bins=[0, 1, 2, 3, 4],
                labels=['Very Short', 'Short', 'Medium', 'Long']
            )
        
        return df_engineered

# ============================================================
# IMPROVED VISUALIZATION FUNCTIONS
# ============================================================
def create_zoomed_visualizations(summary_df, last_pros_scores, last_labels, timestamp):
    """Create zoomed-in visualizations that use full axis height"""
    
    print("\nGenerating ZOOMED visualizations...")
    
    sns.set(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for clarity
    colors = ['#FF5252', '#4CAF50', '#2196F3']  # Red, Green, Blue
    if_color = '#95A5A6'  # Gray for Isolation Forest
    
    # ========== 1. F1-SCORE COMPARISON (ZOOMED) ==========
    ax = axes[0, 0]
    
    # Collect all F1 values to determine zoom range
    all_f1_values = []
    for i, top_k in enumerate(TOP_K_THRESHOLDS):
        mask = (summary_df['model'] == 'PROS') & (summary_df['threshold_top_k'] == top_k)
        all_f1_values.extend(summary_df.loc[mask, 'f1_score'].values)
    
    mask_if = summary_df['model'] == 'IsolationForest'
    all_f1_values.extend(summary_df.loc[mask_if, 'f1_score'].values)
    
    # Calculate zoom range (with 10% padding)
    f1_min = max(0, np.min(all_f1_values) - 0.05)  # Don't go below 0
    f1_max = min(1.0, np.max(all_f1_values) + 0.05)  # Don't go above 1.0
    
    for i, top_k in enumerate(TOP_K_THRESHOLDS):
        mask = (summary_df['model'] == 'PROS') & (summary_df['threshold_top_k'] == top_k)
        ax.plot(summary_df.loc[mask, 'bot_fraction'], 
               summary_df.loc[mask, 'f1_score'], 
               'o-', color=colors[i], linewidth=2, markersize=8,
               label=f'PROS Top-{int(top_k*100)}%')
    
    ax.plot(summary_df.loc[mask_if, 'bot_fraction'], 
           summary_df.loc[mask_if, 'f1_score'], 
           's--', color=if_color, linewidth=2, markersize=8,
           label='Isolation Forest')
    
    ax.set_xlabel('Bot Fraction', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score Comparison (Zoomed)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_ylim([f1_min, f1_max])  # ZOOMED Y-AXIS
    
    # ========== 2. PRECISION COMPARISON (ZOOMED) ==========
    ax = axes[0, 1]
    
    # Calculate zoom range for precision
    all_precision_values = []
    for i, top_k in enumerate(TOP_K_THRESHOLDS):
        mask = (summary_df['model'] == 'PROS') & (summary_df['threshold_top_k'] == top_k)
        all_precision_values.extend(summary_df.loc[mask, 'precision'].values)
    all_precision_values.extend(summary_df.loc[mask_if, 'precision'].values)
    
    precision_min = max(0, np.min(all_precision_values) - 0.05)
    precision_max = min(1.0, np.max(all_precision_values) + 0.05)
    
    for i, top_k in enumerate(TOP_K_THRESHOLDS):
        mask = (summary_df['model'] == 'PROS') & (summary_df['threshold_top_k'] == top_k)
        ax.plot(summary_df.loc[mask, 'bot_fraction'], 
               summary_df.loc[mask, 'precision'], 
               'o-', color=colors[i], linewidth=2, markersize=8)
    
    ax.plot(summary_df.loc[mask_if, 'bot_fraction'], 
           summary_df.loc[mask_if, 'precision'], 
           's--', color=if_color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Bot Fraction', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision Comparison (Zoomed)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([precision_min, precision_max])  # ZOOMED Y-AXIS
    
    # ========== 3. RECALL COMPARISON (ZOOMED) ==========
    ax = axes[1, 0]
    
    # Calculate zoom range for recall
    all_recall_values = []
    for i, top_k in enumerate(TOP_K_THRESHOLDS):
        mask = (summary_df['model'] == 'PROS') & (summary_df['threshold_top_k'] == top_k)
        all_recall_values.extend(summary_df.loc[mask, 'recall'].values)
    all_recall_values.extend(summary_df.loc[mask_if, 'recall'].values)
    
    recall_min = max(0, np.min(all_recall_values) - 0.05)
    recall_max = min(1.0, np.max(all_recall_values) + 0.05)
    
    for i, top_k in enumerate(TOP_K_THRESHOLDS):
        mask = (summary_df['model'] == 'PROS') & (summary_df['threshold_top_k'] == top_k)
        ax.plot(summary_df.loc[mask, 'bot_fraction'], 
               summary_df.loc[mask, 'recall'], 
               'o-', color=colors[i], linewidth=2, markersize=8)
    
    ax.plot(summary_df.loc[mask_if, 'bot_fraction'], 
           summary_df.loc[mask_if, 'recall'], 
           's--', color=if_color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Bot Fraction', fontsize=11)
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Recall Comparison (Zoomed)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([recall_min, recall_max])  # ZOOMED Y-AXIS
    
    # ========== 4. SCORE DISTRIBUTION (IMPROVED) ==========
    ax = axes[1, 1]
    
    # Separate scores by label
    bot_scores = last_pros_scores[last_labels == 1]
    human_scores = last_pros_scores[last_labels == 0]
    
    # Plot histograms with better visualization
    n_bins = 40  # More bins for better resolution
    
    # Human scores (blue)
    human_counts, human_bins, _ = ax.hist(human_scores, bins=n_bins, alpha=0.6, 
                                         color='#3498DB', edgecolor='black', 
                                         linewidth=0.5, label='Human', density=True)
    
    # Bot scores (red) - TRANSPARENT overlay
    bot_counts, bot_bins, _ = ax.hist(bot_scores, bins=n_bins, alpha=0.6, 
                                     color='#E74C3C', edgecolor='black', 
                                     linewidth=0.5, label='Bot', density=True)
    
    # Calculate thresholds and ensure visibility
    thresholds = []
    threshold_colors = []
    threshold_labels = []
    
    for i, top_k in enumerate(TOP_K_THRESHOLDS):
        threshold = np.quantile(last_pros_scores, 1 - top_k)
        thresholds.append(threshold)
        threshold_colors.append(colors[i])
        threshold_labels.append(f'Top-{int(top_k*100)}% (Score={threshold:.2f})')
    
    # Sort thresholds to avoid overlap
    sorted_indices = np.argsort(thresholds)
    
    # Plot thresholds with OFFSET to ensure visibility
    y_max = max(np.max(human_counts), np.max(bot_counts))
    for idx in sorted_indices:
        threshold = thresholds[idx]
        color = threshold_colors[idx]
        label = threshold_labels[idx]
        
        # Calculate offset for line position
        offset = 0.02 * y_max * (idx + 1)
        
        # Draw the threshold line with bold style
        ax.axvline(threshold, color=color, linestyle='--', linewidth=2.5, 
                  label=label, ymax=0.95 - (idx * 0.05))  # Adjust vertical position
    
    ax.set_xlabel('PROS Anomaly Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Score Distribution (Bot={BOT_FRACTIONS[-1]:.0%})', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Improved legend with better positioning
    ax.legend(fontsize=8, loc='upper right', 
             bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    
    # Add statistical annotations
    separation = np.mean(bot_scores) - np.mean(human_scores)
    
    # FIXED: Calculate overlap properly
    # Create combined distribution and calculate overlap area
    overlap_threshold = 0.1
    # Calculate overlap by checking how many scores from each distribution fall near the other's mean
    bot_near_human_mean = np.sum(np.abs(bot_scores - np.mean(human_scores)) < overlap_threshold)
    human_near_bot_mean = np.sum(np.abs(human_scores - np.mean(bot_scores)) < overlap_threshold)
    overlap_count = bot_near_human_mean + human_near_bot_mean
    overlap_percentage = overlap_count / (len(bot_scores) + len(human_scores))
    
    stats_text = (f'Separation: {separation:.2f}\n'
                  f'Human μ: {np.mean(human_scores):.2f}\n'
                  f'Bot μ: {np.mean(bot_scores):.2f}\n'
                  f'Overlap: {overlap_percentage:.1%}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(YOUTUBE_RESULTS_DIR, f"synthetic_zoomed_visualization_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"ZOOMED visualization saved: {plot_path}")
    
    return plot_path

# ============================================================
# MAIN SCRIPT WITH IMPROVEMENTS
# ============================================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TERMINAL_OUTPUT_PATH = os.path.join(YOUTUBE_RESULTS_DIR, f"synthetic_improved_output_{timestamp}.txt")

    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8", buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def close(self):
            self.log.close()

    logger = Logger(TERMINAL_OUTPUT_PATH)
    sys.stdout = logger

    try:
        print("=" * 70)
        print("IMPROVED SYNTHETIC VALIDATION - PROS BOT DETECTION")
        print("=" * 70)
        print(f"Data: {N_CHANNELS} channels × {COMMENTS_PER_CHANNEL} comments = {N_CHANNELS*COMMENTS_PER_CHANNEL:,} total")
        print(f"Bot fractions: {BOT_FRACTIONS}")
        print(f"Top-K thresholds: {TOP_K_THRESHOLDS}")
        print("=" * 70)
        
        summary_rows = []
        last_pros_scores = None
        last_labels = None
        
        # Use IMPROVED detector
        pros = SyntheticPROSDetector(min_samples_per_bin=10, jsd_threshold=0.12)
        
        for bot_frac in BOT_FRACTIONS:
            print(f"\n{'='*40}")
            print(f"Bot fraction = {bot_frac:.0%}")
            print(f"{'='*40}")
            
            # Generate IMPROVED synthetic data
            df = generate_realistic_synthetic_youtube_data(bot_fraction=bot_frac)
            print(f"Generated: {len(df):,} comments ({df['label'].sum():,} bots = {df['label'].mean():.1%})")
            
            # Run improved PROS analysis
            results = pros.run_full_analysis(df)
            df_scored = results['df_scored']
            df_scored['label'] = df['label'].values
            
            # Store for visualization
            if bot_frac == BOT_FRACTIONS[-1]:
                last_pros_scores = df_scored['pros_anomaly_score'].values.copy()
                last_labels = df_scored['label'].values.copy()
            
            # Calculate metrics
            scores = df_scored['pros_anomaly_score'].values
            labels = df_scored['label'].values
            
            for top_k in TOP_K_THRESHOLDS:
                threshold = np.quantile(scores, 1 - top_k)
                preds = (scores >= threshold).astype(np.int8)
                
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                summary_rows.append({
                    'bot_fraction': bot_frac,
                    'threshold_top_k': top_k,
                    'model': 'PROS',
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4),
                    'tp': tp, 'fp': fp, 'fn': fn
                })
                
                print(f"[PROS] Top-{int(top_k*100)}% | P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
            
            # Isolation Forest baseline
            df_scored = pros.compare_with_isolation_forest(df_scored)
            preds_if = df_scored['iso_forest_anomaly'].values.astype(np.int8)
            
            tp_if = np.sum((preds_if == 1) & (labels == 1))
            fp_if = np.sum((preds_if == 1) & (labels == 0))
            fn_if = np.sum((preds_if == 0) & (labels == 1))
            
            precision_if = tp_if / (tp_if + fp_if) if (tp_if + fp_if) > 0 else 0
            recall_if = tp_if / (tp_if + fn_if) if (tp_if + fn_if) > 0 else 0
            f1_if = 2 * precision_if * recall_if / (precision_if + recall_if) if (precision_if + recall_if) > 0 else 0
            
            summary_rows.append({
                'bot_fraction': bot_frac,
                'threshold_top_k': 'auto',
                'model': 'IsolationForest',
                'precision': round(precision_if, 4),
                'recall': round(recall_if, 4),
                'f1_score': round(f1_if, 4),
                'tp': tp_if, 'fp': fp_if, 'fn': fn_if
            })
            
            print(f"[IsolationForest] P={precision_if:.3f} R={recall_if:.3f} F1={f1_if:.3f}")
        
        # Save results
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(YOUTUBE_RESULTS_DIR, f"synthetic_improved_summary_{timestamp}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n✓ Summary saved: {summary_csv}")
        
        # ============================================================
        # CREATE IMPROVED VISUALIZATIONS
        # ============================================================
        if last_pros_scores is not None and last_labels is not None:
            print("\n" + "="*70)
            print("CREATING ZOOMED VISUALIZATIONS")
            print("="*70)
            
            # Create zoomed visualizations
            plot_path = create_zoomed_visualizations(summary_df, last_pros_scores, last_labels, timestamp)
            
            # Calculate and display key metrics
            bot_scores = last_pros_scores[last_labels == 1]
            human_scores = last_pros_scores[last_labels == 0]
            
            bot_mean = np.mean(bot_scores)
            human_mean = np.mean(human_scores)
            separation = bot_mean - human_mean
            
            print(f"\nSCORE DISTRIBUTION ANALYSIS (30% bots):")
            print(f"  Human scores: μ={human_mean:.3f}, σ={np.std(human_scores):.3f}, range=[{np.min(human_scores):.3f}, {np.max(human_scores):.3f}]")
            print(f"  Bot scores: μ={bot_mean:.3f}, σ={np.std(bot_scores):.3f}, range=[{np.min(bot_scores):.3f}, {np.max(bot_scores):.3f}]")
            print(f"  Separation: {separation:.3f} (Higher = better discrimination)")
            
            # Calculate overlap metric
            overlap_threshold = 0.1  # Scores within 0.1 are considered overlapping
            overlap_count = np.sum(np.abs(bot_scores - human_mean) < overlap_threshold) + \
                          np.sum(np.abs(human_scores - bot_mean) < overlap_threshold)
            overlap_percentage = overlap_count / (len(bot_scores) + len(human_scores))
            print(f"  Overlap (<{overlap_threshold}): {overlap_percentage:.1%} (Lower = better)")
            
            # Calculate thresholds
            print(f"\nDETECTION THRESHOLDS (30% bots):")
            for top_k in TOP_K_THRESHOLDS:
                threshold = np.quantile(last_pros_scores, 1 - top_k)
                bots_above = np.sum(bot_scores >= threshold)
                humans_above = np.sum(human_scores >= threshold)
                print(f"  Top-{int(top_k*100)}%: Score ≥ {threshold:.3f} "
                      f"({bots_above:,}/{len(bot_scores):,} bots = {bots_above/len(bot_scores):.1%}, "
                      f"{humans_above:,}/{len(human_scores):,} humans = {humans_above/len(human_scores):.1%})")
            
            # Calculate AUC approximation
            print(f"\nPERFORMANCE METRICS:")
            
            # For each threshold, calculate actual metrics
            for top_k in TOP_K_THRESHOLDS:
                threshold = np.quantile(last_pros_scores, 1 - top_k)
                preds = (last_pros_scores >= threshold).astype(int)
                
                tp = np.sum((preds == 1) & (last_labels == 1))
                fp = np.sum((preds == 1) & (last_labels == 0))
                fn = np.sum((preds == 0) & (last_labels == 1))
                tn = np.sum((preds == 0) & (last_labels == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"  Top-{int(top_k*100)}%: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, "
                      f"TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")
        
        print("\n" + "="*70)
        print("IMPROVED SYNTHETIC VALIDATION COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"\n✓ Log saved: {TERMINAL_OUTPUT_PATH}")