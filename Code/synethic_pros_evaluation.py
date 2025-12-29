import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from Youtube_model import PROSDetector
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "synthetic_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
BOT_FRACTIONS = [0.05, 0.10, 0.20, 0.30]      # attack strength
TOP_K_THRESHOLDS = [0.05, 0.10, 0.15]         # decision thresholds
N_CHANNELS = 1000
COMMENTS_PER_CHANNEL = 5

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_data(
    n_channels=1000,
    comments_per_channel=5,
    bot_fraction=0.1,
    random_seed=42
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
                'isDuplicateComment': is_duplicate,
                'label': int(is_bot_channel)   # Ground truth
            })

    return pd.DataFrame(data)

# ============================================================
# RUN EXPERIMENTS
# ============================================================

summary_rows = []

for bot_frac in BOT_FRACTIONS:
    print("\n" + "=" * 70)
    print(f"Running experiment with bot fraction = {bot_frac:.0%}")
    print("=" * 70)

    df = generate_synthetic_data(
        n_channels=N_CHANNELS,
        comments_per_channel=COMMENTS_PER_CHANNEL,
        bot_fraction=bot_frac,
        random_seed=RANDOM_SEED
    )

    pros = PROSDetector(min_samples_per_bin=5, kl_threshold=0.05)
    results = pros.run_full_analysis(df)
    df_scored = results['df_scored']
    df_scored['label'] = df['label'].values

    # --------------------------------------------------------
    # Evaluate at multiple thresholds
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

    # Save per-scenario detailed results
    scenario_path = os.path.join(
        RESULTS_DIR,
        f"synthetic_results_botfrac_{int(bot_frac*100)}.csv"
    )
    df_scored.to_csv(scenario_path, index=False)

# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(RESULTS_DIR, "synthetic_summary_metrics.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 70)
print("All experiments completed.")
print(f"Summary saved to: {summary_csv}")
print("=" * 70)
print(summary_df)
