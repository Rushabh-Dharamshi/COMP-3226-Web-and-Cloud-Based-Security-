# COMP-3226-Web-and-Cloud-Based-Security-
# Table of Contents

1. [Introduction](#introduction)
2. [Web Log PROS Analysis - Setup & Execution Guide](#web-log-pros-analysis---setup--execution-guide)
   * [Prerequisites](#prerequisites)
   * [Step 1: Download & Organize Logs](#step-1-download--organize-logs)
   * [Step 2: Parse & Feature Engineering](#step-2-parse--feature-engineering)
   * [Step 3: Train Clean Distributions (Algorithm 1)](#step-3-train-clean-distributions-algorithm-1)
   * [Step 4: Run PROS Detection & Evaluation (Algorithm 2)](#step-4-run-pros-detection--evaluation-algorithm-2)
   * [Troubleshooting](#troubleshooting)
3. [Youtube Comments Analysis - Setup & Execution Guide](#youtube-comments-analysis---setup--execution-guide)
<<<<<<< HEAD
=======
   * [Data Extraction](#data-extraction)
   * [Prequisites - Delete Youtube Results folder](#prequisite-delete-youtube-results-folder)
   * [Step 1: Run PROS Analysis on Youtube Comments](#step-1-run-pros-analysis-on-youtube-comments)
   * [Step 2: Run Synthetic Data Evaluation](#step-2-run-synthetic-data-evaluation)
>>>>>>> b1f542c (Update README.md with setup instructions)

# Introduction
This README.md provides the step-by-step instructions needed to run the web log and YouTube comment analysis project. The purpose of the weblog dataset is to prove that the PROS algoritihm works based on ground truth labels. 

# Web Log PROS Analysis - Setup & Execution Guide

This repository contains the implementation of the PROS (Probabilistic Reasoning for Outlier Selection) algorithm for detecting bot traffic in web server logs.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following Python libraries:

```bash
pip install pandas numpy scipy scikit-learn matplotlib user-agents geoip2
```
**Important: You also need the GeoLite2-City.mmdb database for IP geolocation (Cuurently provided in the directory. If that does not work follow, the steps below to install GeoLite2).**

1. Sign up for a free account at [MaxMind](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data/).
2. Download the GeoLite2 City database (MMDB format).
3. Place GeoLite2-City.mmdb in the root directory of this project.

## Step 1: Download & Organize Logs
These scripts download the raw .gz logs dated from January 2015 to latest date from secrepo.com and organize them into folders.

Run the downloader script:

 ```Bash

python web_log_dataset.py
```
**If for whatever reason the script doesn't work the alternative method to collect data is using [wget](https://eternallybored.org/misc/wget/):**

Steps to install wget:

1. Copy the downloaded wget.exe file.
2. Navigate to C:\Windows\System32.
3. Paste the file there. (You will need administrator privileges to do this).
4. Verification. Open Command Prompt (cmd) and type wget --version. If installed correctly, it will display the version information.

**Note: Placing it in System32 automatically adds it to your system path, allowing you to run it from any command prompt window.**

Run this one-line command:
```bash 
wget -U "Mozilla/5.0" -r -np -nH --cut-dirs=1 -R "index.html*" https://www.secrepo.com/self.logs/
```

Output: A folder with web logs from 2015 till date

Run the organiser script to sort them into all_access_logs and all_error_logs folders:
```Bash
python web_log_organiser.py
```
Output: A folder structure (with 2 folders, one for access logs and the other for error logs) containing the raw log files.

## Step 2: Parse & Feature Engineering
This step contains 2 scripts:
1. The dataframer script converts the raw text logs into a structured CSV file with the extracted recorded field (IP address, HTTP status, timestamp, etc.)
2. the labeller script uses the recorded fields to obtain the features that are used for buckets (Browser, OsFamily, state, etc.) and then labels the logs if they are a bot or not

**Check Paths: Update the log_folder path in the script to match where your logs are stored.**

Run the dataframer script:

```Bash
python web_log_data.py
```
Output: web_log_data.csv (This dataset will then be applied to the labeller script).

Run the labeller script:

```Bash
python web_log_labeller.py
```
Output: processed_web_log_features.csv (The main dataset used for analysis).

## Step 3: Train Clean Distributions (Algorithm 1)
This script learns the "normal" behavior baselines by finding unattacked buckets of traffic (e.g., finding the stable distribution of Chrome versions within US traffic).

Run the Algorithm 1 script:

```Bash
python web_log_PROS_algo1.py
```
Output: A clean_distributions/ folder containing CSV files of probability vectors based on target feature and conditional feature (e.g., clean_browser_by_family.csv).

## Step 4: Run PROS Detection & Evaluation (Algorithm 2)

This final script calculates anomaly scores for every request, trains the baseline Isolation Forest, and generates the evaluation plots for comparison.

Run the Algorithm 2 script:

```Bash
python web_log_PROS_algo2.py
```
Output:
1. Scored_traffic.csv: The full dataset sorted by bot_odds (suspiciousness score). The highest scores are at the top.
2. roc_curve_comparison.png: A graph comparing PROS vs. Isolation Forest accuracy.
3. Console Output: Precision/Recall/AUC metrics and a preview of the most suspicious requests.

## Troubleshooting

1. Memory Error: Step 4 crashes with a MemoryError or ArrayMemoryError if all the data is used for one hot encoding due to a dense matrix so it is very easy for computers to run out of RAM. Due to this Isolation Forest only requires 100,000 rows, which is statistically sufficient to generate the ROC curve (feel free to change the sample size based on your RAM capacity).

2. Missing GeoIP: If geolocation columns show "Unknown", ensure GeoLite2-City.mmdb is in the correct folder.

3. Slow Processing: Step 2 may take 5-10 minutes depending on your CPU speed as it processes millions of log lines.

# Youtube Comments Analysis - Setup & Execution Guide

## Data Extraction

- All the data has been collected for us in Code/Youtube Data/ folder. You do not need to run extraction scripts if the files exist.

Folder structure:

Youtube Data/
├── video_ids.xlsx
├── Youtube_extracted_data.csv
├── Youtube_extracted_data.json
└── Youtube_extracted_data_append.json

Notes:

video_ids.xlsx → Input video IDs

Youtube_extracted_data_append.json → Temporary team data

Youtube_extracted_data.json → Final consolidated dataset

## prequisite: Delete Youtube Results folder

- You must delete the Youtube Results folder if you want to run the python files to produce the results


## Step 1: Run PROS Analysis on YouTube Comments

Navigate to the Code/Youtube Code/ folder:

Run the PROS algorithm:

```Bash
python Youtube_model.py
```


Outputs in the Youtube Results/ folder:


- pros_terminal_output_[TIMESTAMP].txt:	Full console log of PROS execution
- pros_visualization_1_rq_overview.png: Overview dashboard (score distribution, feature importance, genre analysis)
- pros_visualization_2_feature_analysis.png: Feature analysis plots (Average Bot Score against account age, posting velocity, profile completeness and a list of top suspicious channels)

- pros_genre_analysis_[TIMESTAMP].csv:	Genre-level bot likelihood statistics
- pros_clean_distributions_[TIMESTAMP].json	Estimated clean distributions for each feature
- pros_rq_metrics_[TIMESTAMP].json	Performance metrics and evaluation results


## Step 2: Run Synthetic Data Evaluation

```Bash
python synthetic_pros_evaluation.py
```
- Generates realistic synthetic YouTube data with known bot/human labels

- Tests PROS performance at different bot fractions (5%, 10%, 20%, 30%)

- Compares PROS against Isolation Forest

- Produces detailed visualizations



Outputs in Youtube Results/ folder:

- synthetic_improved_output_[TIMESTAMP].txt:	Console log of all synthetic tests and performance metrics
- synthetic_improved_summary_[TIMESTAMP].csv:	Performance summary across bot fractions and thresholds
- synthetic_zoomed_visualization_[TIMESTAMP].png:	Four-panel visualization: F1-Score, Precision, Recall, and Score Distribution with stats
