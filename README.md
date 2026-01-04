# COMP-3226-Web-and-Cloud-Based-Security-
This README.md provides the step-by-step instructions needed to run your web log analysis project.

# Web Log PROS Analysis - Setup & Execution Guide

This repository contains the implementation of the PROS (Probabilistic Reasoning for Outlier Selection) algorithm for detecting bot traffic in web server logs.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following Python libraries:

```bash
pip install pandas numpy scipy scikit-learn matplotlib user-agents geoip2
```
**Important: You also need the GeoLite2-City.mmdb database for IP geolocation (Cuurently provided in the directory, if that does work follow the steps below to install GeoLite2.**

1. Sign up for a free account at MaxMind.
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
2. the labeller script uses the recorded fields to obtain the deatures that are used for buckets (Browser, OsFamily, state, etc.) and then labels the logs if they are a bot or not

Check Paths: Update the log_folder path in the script to match where your logs are stored.

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
Output: A clean_distributions/ folder containing CSV files of probability vectors based on taget feature and conditional feature (e.g., clean_browser_by_family.csv).

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

1. Memory Error: Step 4 crashes with a MemoryError or ArrayMemoryError if all the data is used for one hot encoding due to a dense matrix so it is very easy for computers to run out of RAM. Due to this Isolation Forest only requires 100,000 rows is statistically sufficient to generate the ROC curve (feel free to change the sample size based on your RAM capacity).

2. Missing GeoIP: If geolocation columns show "Unknown", ensure GeoLite2-City.mmdb is in the correct folder.

3. Slow Processing: Step 2 may take 5-10 minutes depending on your CPU speed as it processes millions of log lines.
