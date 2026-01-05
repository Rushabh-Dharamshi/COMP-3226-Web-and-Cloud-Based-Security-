\# COMP-3226-Web-and-Cloud-Based-Security-

\# Table of Contents



1\. \[Introduction](#introduction)

2\. \[Web Log PROS Analysis - Setup \& Execution Guide](#web-log-pros-analysis---setup--execution-guide)

&nbsp;  \* \[Prerequisites](#prerequisites)

&nbsp;  \* \[Step 1: Download \& Organize Logs](#step-1-download--organize-logs)

&nbsp;  \* \[Step 2: Parse \& Feature Engineering](#step-2-parse--feature-engineering)

&nbsp;  \* \[Step 3: Train Clean Distributions (Algorithm 1)](#step-3-train-clean-distributions-algorithm-1)

&nbsp;  \* \[Step 4: Run PROS Detection \& Evaluation (Algorithm 2)](#step-4-run-pros-detection--evaluation-algorithm-2)

&nbsp;  \* \[Troubleshooting](#troubleshooting)

3\. \[Youtube Comments Analysis - Setup \& Execution Guide](#youtube-comments-analysis---setup--execution-guide)



\# Introduction

This README.md provides the step-by-step instructions needed to run the web log and YouTube comment analysis project. The purpose of the weblog dataset is to prove that the PROS algoritihm works based on ground truth labels. 



\# Web Log PROS Analysis - Setup \& Execution Guide



This repository contains the implementation of the PROS (Probabilistic Reasoning for Outlier Selection) algorithm for detecting bot traffic in web server logs.



\## Prerequisites



Ensure you have Python 3.8+ installed. You will need the following Python libraries:



```bash

pip install pandas numpy scipy scikit-learn matplotlib user-agents geoip2

```

\*\*Important: You also need the GeoLite2-City.mmdb database for IP geolocation (Cuurently provided in the directory. If that does not work follow, the steps below to install GeoLite2).\*\*



1\. Sign up for a free account at \[MaxMind](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data/).

2\. Download the GeoLite2 City database (MMDB format).

3\. Place GeoLite2-City.mmdb in the root directory of this project.



\## Step 1: Download \& Organize Logs

These scripts download the raw .gz logs dated from January 2015 to latest date from secrepo.com and organize them into folders.



Run the downloader script:



&nbsp;```Bash



python web\_log\_dataset.py

```

\*\*If for whatever reason the script doesn't work the alternative method to collect data is using \[wget](https://eternallybored.org/misc/wget/):\*\*



Steps to install wget:



1\. Copy the downloaded wget.exe file.

2\. Navigate to C:\\Windows\\System32.

3\. Paste the file there. (You will need administrator privileges to do this).

4\. Verification. Open Command Prompt (cmd) and type wget --version. If installed correctly, it will display the version information.



\*\*Note: Placing it in System32 automatically adds it to your system path, allowing you to run it from any command prompt window.\*\*



Run this one-line command:

```bash 

wget -U "Mozilla/5.0" -r -np -nH --cut-dirs=1 -R "index.html\*" https://www.secrepo.com/self.logs/

```



Output: A folder with web logs from 2015 till date



Run the organiser script to sort them into all\_access\_logs and all\_error\_logs folders:

```Bash

python web\_log\_organiser.py

```

Output: A folder structure (with 2 folders, one for access logs and the other for error logs) containing the raw log files.



\## Step 2: Parse \& Feature Engineering

This step contains 2 scripts:

1\. The dataframer script converts the raw text logs into a structured CSV file with the extracted recorded field (IP address, HTTP status, timestamp, etc.)

2\. the labeller script uses the recorded fields to obtain the features that are used for buckets (Browser, OsFamily, state, etc.) and then labels the logs if they are a bot or not



\*\*Check Paths: Update the log\_folder path in the script to match where your logs are stored.\*\*



Run the dataframer script:



```Bash

python web\_log\_data.py

```

Output: web\_log\_data.csv (This dataset will then be applied to the labeller script).



Run the labeller script:



```Bash

python web\_log\_labeller.py

```

Output: processed\_web\_log\_features.csv (The main dataset used for analysis).



\## Step 3: Train Clean Distributions (Algorithm 1)

This script learns the "normal" behavior baselines by finding unattacked buckets of traffic (e.g., finding the stable distribution of Chrome versions within US traffic).



Run the Algorithm 1 script:



```Bash

python web\_log\_PROS\_algo1.py

```

Output: A clean\_distributions/ folder containing CSV files of probability vectors based on target feature and conditional feature (e.g., clean\_browser\_by\_family.csv).



\## Step 4: Run PROS Detection \& Evaluation (Algorithm 2)



This final script calculates anomaly scores for every request, trains the baseline Isolation Forest, and generates the evaluation plots for comparison.



Run the Algorithm 2 script:



```Bash

python web\_log\_PROS\_algo2.py

```

Output:

1\. Scored\_traffic.csv: The full dataset sorted by bot\_odds (suspiciousness score). The highest scores are at the top.

2\. roc\_curve\_comparison.png: A graph comparing PROS vs. Isolation Forest accuracy.

3\. Console Output: Precision/Recall/AUC metrics and a preview of the most suspicious requests.



\## Troubleshooting



1\. Memory Error: Step 4 crashes with a MemoryError or ArrayMemoryError if all the data is used for one hot encoding due to a dense matrix so it is very easy for computers to run out of RAM. Due to this Isolation Forest only requires 100,000 rows, which is statistically sufficient to generate the ROC curve (feel free to change the sample size based on your RAM capacity).



2\. Missing GeoIP: If geolocation columns show "Unknown", ensure GeoLite2-City.mmdb is in the correct folder.



3\. Slow Processing: Step 2 may take 5-10 minutes depending on your CPU speed as it processes millions of log lines.



\# Youtube Comments Analysis - Setup \& Execution Guide



This section explains how to run the PROS algorithm and synthetic validation for YouTube comment data. The goal is to detect bot comments and evaluate the algorithm’s performance with both real and synthetic datasets.



Prerequisites



Ensure you have Python 3.8+ installed. Install the required Python libraries:



pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl





Optional: For reproducibility, set up a virtual environment:



python -m venv venv\_youtube

source venv\_youtube/bin/activate  # Linux / Mac

venv\_youtube\\Scripts\\activate     # Windows



Step 1: Prepare Dataset



All the required data is already provided in the Youtube Data/ folder.

You do not need to run extraction scripts if the files exist.



Folder structure:



Youtube Data/

├── video\_ids.xlsx

├── Youtube\_extracted\_data.csv

├── Youtube\_extracted\_data.json

└── Youtube\_extracted\_data\_append.json





Notes:



video\_ids.xlsx → Input video IDs



Youtube\_extracted\_data\_append.json → Temporary team data



Youtube\_extracted\_data.json → Final consolidated dataset



If you want to merge team data:



\# merge\_json\_files.py will automatically merge the JSON files

python merge\_json\_files.py



Step 2: Run PROS Analysis on YouTube Comments



Navigate to the Youtube Code/ folder:



cd Code/Youtube\\ Code/



Run PROS Algorithm

python Youtube\_model.py





Outputs in Youtube Results/ folder:



File	Description

pros\_terminal\_output\_\[TIMESTAMP].txt	Full console log of PROS execution

pros\_visualization\_1\_rq\_overview.png	Overview dashboard (score distribution, feature importance, genre analysis)

pros\_visualization\_2\_feature\_analysis.png	Feature analysis plots (account age, posting velocity, profile completeness)

pros\_genre\_analysis\_\[TIMESTAMP].csv	Genre-level bot likelihood statistics

pros\_clean\_distributions\_\[TIMESTAMP].json	Estimated clean distributions for each feature

pros\_rq\_metrics\_\[TIMESTAMP].json	Performance metrics and evaluation results

Step 3: Run Synthetic Data Evaluation

python synthetic\_pros\_evaluation.py





This script:



Generates realistic synthetic YouTube data with known bot/human labels



Tests PROS performance at different bot fractions (5%, 10%, 20%, 30%)



Compares PROS against Isolation Forest



Produces detailed visualizations



Outputs in Youtube Results/ folder:



File	Description

synthetic\_improved\_output\_\[TIMESTAMP].txt	Console log of all synthetic tests and performance metrics

synthetic\_improved\_summary\_\[TIMESTAMP].csv	Performance summary across bot fractions and thresholds

synthetic\_zoomed\_visualization\_\[TIMESTAMP].png	Four-panel visualization: F1-Score, Precision, Recall, and Score Distribution with stats

Step 4: Notes \& Recommendations



Clear Previous Results: Remove old files in Youtube Results/ to avoid confusion.



Execution Time: Youtube\_model.py may take several minutes depending on dataset size.



Synthetic Evaluation: Provides a controlled environment to validate algorithm performance.



Custom Thresholds: Modify TOP\_K\_THRESHOLDS or bot fractions in synthetic\_pros\_evaluation.py for experimentation.



This section aligns perfectly with the Web Log section style and keeps everything GitHub-friendly and readable.



