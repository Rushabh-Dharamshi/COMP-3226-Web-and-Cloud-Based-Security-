import pandas as pd
import gzip
import glob
import re
import os

# 1. SETUP: Point to your folder
# Make sure this path matches where you put the logs
log_folder = r"D:\cloud and web based security\Project\self.logs\all_access_logs"
log_files = glob.glob(os.path.join(log_folder, "*.gz"))

print(f"Found {len(log_files)} log files.")

# 2. DEFINE PATTERN: Apache Combined Log Format Regex
LOG_PATTERN = re.compile(
    r'(?P<ip>[\d\.]+) - - \[(?P<timestamp>.*?)\] "(?P<request>.*?)" (?P<status>\d+) (?P<size>\d+|-) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
)

def parse_log_line(line):
    match = LOG_PATTERN.match(line)
    if match:
        return match.groupdict()
    return None

# 3. READ DATA
data = []
# NOTE: For testing, we are reading only the first 5 files. 
# To read ALL files, remove "[:5]" from the loop below (e.g., "for file_path in log_files:")
print("Parsing logs (this may take a moment)...")
count = 0
total_files = len(log_files)

for file_path in log_files:
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parsed = parse_log_line(line)
                if parsed:
                    data.append(parsed)
        
        # Simple progress indicator
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{total_files} files...")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print(f"Finished parsing. Total records found: {len(data)}")

# 4. CONVERT TO DATAFRAME
df = pd.DataFrame(data)

# 5. FEATURE EXTRACTION (As per paper)
# Split the request into Method, Path, Protocol
# e.g., "GET /index.html HTTP/1.1" -> "GET", "/index.html", "HTTP/1.1"
if not df.empty:
    request_split = df['request'].str.split(' ', n=2, expand=True)
    # Handle cases where split might fail (less than 3 parts)
    if request_split.shape[1] == 3:
        df[['method', 'path', 'protocol']] = request_split
    else:
        print("Warning: Some requests could not be split perfectly. Check data quality.")

# 6. SAVE TO CSV
output_filename = "web_log_data.csv"
print(f"\nSaving {len(df)} rows to '{output_filename}'...")

# index=False prevents pandas from adding an extra row number column
df.to_csv(output_filename, index=False)

print(f"Done! File saved to: {os.path.abspath(output_filename)}")