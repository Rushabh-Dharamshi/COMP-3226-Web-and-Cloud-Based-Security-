# COMP-3226-Web-and-Cloud-Based-Security-
This README.md provides the step-by-step instructions needed to run your web log analysis project.

Markdown

# Web Log PROS Analysis - Setup & Execution Guide

This repository contains the implementation of the PROS (Probabilistic Reasoning for Outlier Selection) algorithm for detecting bot traffic in web server logs.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following Python libraries:

```bash
pip install pandas numpy scipy scikit-learn matplotlib user-agents geoip2
Important: You also need the GeoLite2-City.mmdb database for IP geolocation.
```
Sign up for a free account at MaxMind.

Download the GeoLite2 City database (MMDB format).

Place GeoLite2-City.mmdb in the root directory of this project.

Step 1: Download & Organize Logs
These scripts download the raw .gz logs from secrepo.com and organize them into folders.

Run the downloader script:

Bash

python 1_download_logs.py
Run the organizer script to sort them into all_access_logs and all_error_logs:

Bash

python 2_organize_logs.py
Output: A folder structure containing your raw log files.

Step 2: Parse & Feature Engineering
This step converts the raw text logs into a structured CSV file with extracted features (Browser, OS, Geolocation, etc.).

Check Paths: Update the log_folder path in the script to match where your logs are stored.

Run the parsing script:

Bash

python 3_parse_and_extract.py
Output: processed_web_log_features.csv (The main dataset used for analysis).

Step 3: Train Clean Distributions (Algorithm 1)
This script learns the "normal" behavior baselines by finding unattacked buckets of traffic (e.g., finding the stable distribution of Chrome versions within US traffic).

Run the training script:

Bash

python 4_train_distributions.py
Output: A clean_distributions/ folder containing CSV files of probability vectors (e.g., clean_browser_by_family.csv).

Step 4: Run PROS Detection & Evaluation (Algorithm 2)
This final script calculates anomaly scores for every request, trains the baseline Isolation Forest, and generates the evaluation plots.

Run the analysis script:

Bash

python 5_run_analysis.py
Output:

scored_traffic.csv: The full dataset sorted by bot_odds (suspiciousness score). The highest scores are at the top.

roc_curve_comparison.png: A graph comparing PROS vs. Isolation Forest accuracy.

Console Output: Precision/Recall/AUC metrics and a preview of the most suspicious requests.

Troubleshooting
Memory Error: If Step 4 crashes with a MemoryError or ArrayMemoryError, confirm you are using the optimized version of the script (provided in our chat) which uses sparse matrices and data sampling for the Isolation Forest baseline.

Missing GeoIP: If geolocation columns show "Unknown", ensure GeoLite2-City.mmdb is in the correct folder.

Slow Processing: Step 2 may take 5-10 minutes depending on your CPU speed as it processes millions of log lines.
