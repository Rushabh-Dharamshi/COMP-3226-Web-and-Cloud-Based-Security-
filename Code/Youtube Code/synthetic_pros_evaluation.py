import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from Youtube_model import PROSDetector
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)   
YOUTUBE_RESULTS_DIR = os.path.join(PARENT_DIR, "Youtube Results")
os.makedirs(YOUTUBE_RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
BOT_FRACTIONS = [0.05, 0.10, 0.20, 0.30]  # Proportion of bots
TOP_K_THRESHOLDS = [0.05, 0.10, 0.15]     # Top-k decision thresholds
N_CHANNELS = 1000
COMMENTS_PER_CHANNEL = 5

# ================== TERMINAL LOGGER ========================
TERMINAL_OUTPUT_PATH = os.path.join(YOUTUBE_RESULTS_DIR, "synthetic_terminal_output.txt")

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

sys.stdout = Logger(TERMINAL_OUTPUT_PATH)

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_youtube_data(
    n_channels=N_CHANNELS,
    comments_per_channel=COMMENTS_PER_CHANNEL,
    bot_fraction=0.1,
    random_seed=RANDOM_SEED
):
    np.random.seed(random_seed)
    data = []

    genres = ['Entertainment', 'Gaming', 'Music', 'News & Politics',
              'Howto & Style', 'Sports', 'Education']
    countries = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', None]

    for c in range(n_channels):
        channel_id = f'CH_{c:04d}'
        channel_title = f'Channel_{c}'
        channel_date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1200))
        subscriber_count = int(np.random.exponential(scale=1000))
        video_count = int(np.random.exponential(scale=50))
        view_count = int(np.random.exponential(scale=10000))
        country = np.random.choice(countries)
        has_desc = np.random.choice([True, False], p=[0.8, 0.2])
        default_pic = np.random.choice([True, False], p=[0.3, 0.7])
        genre = np.random.choice(genres)
        is_bot_channel = np.random.rand() < bot_fraction

        for k in range(comments_per_channel):
            comment_date = channel_date + timedelta(days=np.random.randint(0, 1000))
            like_count = 0 if is_bot_channel else np.random.poisson(lam=2)
            comment_text = f"Spam Comment {k}" if is_bot_channel else f"Normal Comment {k}"
            is_duplicate = int(is_bot_channel and np.random.rand() < 0.5)

            data.append({
                'videoID': f'VID_{c}_{k}',
                'channelID': channel_id,
                'channelTitle': channel_title,
                'channelDate': channel_date.isoformat(),
                'channelViewCount': view_count,
                'channelSubscriberCount': subscriber_count,
                'channelVideoCount': video_count,
                'channelCountry': country,
                'commentDate': comment_date.isoformat(),
                'commentLikeCount': like_count,
                'videoGenre': genre,
                'hasDescription': has_desc,
                'channelDescription': 'Some description' if has_desc else '',
                'defaultProfilePic': default_pic,
                'commentText': comment_text,
                'isDuplicateComment': is_duplicate,
                'label': int(is_bot_channel)
            })

    return pd.DataFrame(data)

# ============================================================
# RUN SYNTHETIC EXPERIMENTS
# ============================================================

summary_rows = []

for bot_frac in BOT_FRACTIONS:
    print("\n" + "="*70)
    print(f"Running experiment with bot fraction = {bot_frac:.0%}")
    print("="*70)

    df = generate_synthetic_youtube_data(bot_fraction=bot_frac)

    # Initialize PROSDetector
    pros = PROSDetector(min_samples_per_bin=5, jsd_threshold=0.15)
    results = pros.run_full_analysis(df)
    df_scored = results['df_scored']
    df_scored['label'] = df['label'].values

    # --------------------------------------------------------
    # Evaluate PROS at multiple top-k thresholds
    # --------------------------------------------------------
    for top_k in TOP_K_THRESHOLDS:
        threshold = df_scored['pros_anomaly_score'].quantile(1 - top_k)
        df_scored['pros_pred'] = (df_scored['pros_anomaly_score'] >= threshold).astype(int)

        p = precision_score(df_scored['label'], df_scored['pros_pred'])
        r = recall_score(df_scored['label'], df_scored['pros_pred'])
        f1 = f1_score(df_scored['label'], df_scored['pros_pred'])

        summary_rows.append({
            'bot_fraction': bot_frac,
            'threshold_top_k': top_k,
            'model': 'PROS',
            'precision': p,
            'recall': r,
            'f1_score': f1
        })

        print(f"[PROS] Top-{int(top_k*100)}% | Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    # --------------------------------------------------------
    # Baseline: Isolation Forest
    # --------------------------------------------------------
    df_scored = pros.compare_with_isolation_forest(df_scored)
    df_scored['iso_pred'] = df_scored['iso_forest_anomaly']

    p = precision_score(df_scored['label'], df_scored['iso_pred'])
    r = recall_score(df_scored['label'], df_scored['iso_pred'])
    f1 = f1_score(df_scored['label'], df_scored['iso_pred'])

    summary_rows.append({
        'bot_fraction': bot_frac,
        'threshold_top_k': 'auto',
        'model': 'IsolationForest',
        'precision': p,
        'recall': r,
        'f1_score': f1
    })

    print(f"[IsolationForest] Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

# ============================================================
# SAVE SUMMARY METRICS
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(YOUTUBE_RESULTS_DIR, "synthetic_summary_metrics.csv")
summary_df.to_csv(summary_csv, index=False)
print("\nAll experiments completed. Summary metrics saved to:", summary_csv)

# ============================================================
# VISUALIZATION
# ============================================================

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. F1 Score vs Bot Fraction
ax = axes[0, 0]
for model in summary_df['model'].unique():
    df_model = summary_df[summary_df['model'] == model]
    if model == 'PROS':
        for top_k in TOP_K_THRESHOLDS:
            df_topk = df_model[df_model['threshold_top_k'] == top_k]
            ax.plot(df_topk['bot_fraction'], df_topk['f1_score'], marker='o', label=f"{model} Top-{int(top_k*100)}%")
    else:
        ax.plot(df_model['bot_fraction'], df_model['f1_score'], marker='s', label=model)
ax.set_xlabel("Bot Fraction")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score vs Bot Fraction")
ax.legend()

# 2. Precision vs Bot Fraction
ax = axes[0, 1]
for model in summary_df['model'].unique():
    df_model = summary_df[summary_df['model'] == model]
    if model == 'PROS':
        for top_k in TOP_K_THRESHOLDS:
            df_topk = df_model[df_model['threshold_top_k'] == top_k]
            ax.plot(df_topk['bot_fraction'], df_topk['precision'], marker='o', label=f"{model} Top-{int(top_k*100)}%")
    else:
        ax.plot(df_model['bot_fraction'], df_model['precision'], marker='s', label=model)
ax.set_xlabel("Bot Fraction")
ax.set_ylabel("Precision")
ax.set_title("Precision vs Bot Fraction")
ax.legend()

# 3. Recall vs Bot Fraction
ax = axes[1, 0]
for model in summary_df['model'].unique():
    df_model = summary_df[summary_df['model'] == model]
    if model == 'PROS':
        for top_k in TOP_K_THRESHOLDS:
            df_topk = df_model[df_model['threshold_top_k'] == top_k]
            ax.plot(df_topk['bot_fraction'], df_topk['recall'], marker='o', label=f"{model} Top-{int(top_k*100)}%")
    else:
        ax.plot(df_model['bot_fraction'], df_model['recall'], marker='s', label=model)
ax.set_xlabel("Bot Fraction")
ax.set_ylabel("Recall")
ax.set_title("Recall vs Bot Fraction")
ax.legend()

# 4. PROS Anomaly Score Distribution for last bot fraction
ax = axes[1, 1]
df_last = generate_synthetic_youtube_data(bot_fraction=BOT_FRACTIONS[-1])
pros_last = PROSDetector(min_samples_per_bin=5, jsd_threshold=0.15)
results_last = pros_last.run_full_analysis(df_last)
scores = results_last['df_scored']['pros_anomaly_score']
sns.histplot(scores, bins=50, kde=True, color='steelblue', ax=ax)
ax.axvline(scores.quantile(1-TOP_K_THRESHOLDS[0]), color='orange', linestyle='--', label=f"Top-{int(TOP_K_THRESHOLDS[0]*100)}%")
ax.axvline(scores.quantile(1-TOP_K_THRESHOLDS[1]), color='green', linestyle='--', label=f"Top-{int(TOP_K_THRESHOLDS[1]*100)}%")
ax.set_xlabel("PROS Anomaly Score")
ax.set_ylabel("Count")
ax.set_title(f"PROS Score Distribution (Bot Fraction={BOT_FRACTIONS[-1]:.0%})")
ax.legend()

plt.tight_layout()
plot_path = os.path.join(YOUTUBE_RESULTS_DIR, "synthetic_comparison_visualizations.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print("All visualizations saved to:", plot_path)

# Close terminal log
sys.stdout.log.close()
