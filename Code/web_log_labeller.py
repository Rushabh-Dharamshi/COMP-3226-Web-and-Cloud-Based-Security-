import pandas as pd
import user_agents
import geoip2.database
from datetime import datetime

# 1. LOAD DATA
print("Loading data...")
df = pd.read_csv("web_log_data.csv")

# ==========================================
# PART A: User Agent Features
# ==========================================
print("Extracting User Agent features...")
def parse_ua(ua_string):
    try: return user_agents.parse(ua_string)
    except: return None

unique_uas = df['user_agent'].unique()
parsed_uas = {ua: parse_ua(ua) for ua in unique_uas}

def get_browser_custom(ua_string):
    ua = parsed_uas.get(ua_string)
    if not ua: return "Unknown"
    return f"{ua.browser.family}{ua.browser.version_string}"

def get_family(ua_string):
    ua = parsed_uas.get(ua_string)
    return ua.browser.family if ua else "Unknown"

def get_os(ua_string):
    ua = parsed_uas.get(ua_string)
    if not ua: return "Unknown"
    return f"{ua.os.family}{ua.os.version_string}"

def get_os_family(ua_string):
    ua = parsed_uas.get(ua_string)
    return ua.os.family if ua else "Unknown"

df['browser'] = df['user_agent'].apply(get_browser_custom)
df['family'] = df['user_agent'].apply(get_family)
df['OS'] = df['user_agent'].apply(get_os)
df['osFamily'] = df['user_agent'].apply(get_os_family)

# ==========================================
# PART B: Timestamp Features
# ==========================================
print("Extracting Time features...")

# 1. Clean the timestamp string (Remove brackets '[' and ']')
#    Crucial: The format '%d/...' expects a number first. If the log has '[10/...', it fails.
df['timestamp'] = df['timestamp'].astype(str).str.strip('[]')

# 2. Convert to datetime
#    We use utc=True to fix the "mixed time zones" warning and ensure stability.
df['dt_obj'] = pd.to_datetime(
    df['timestamp'], 
    format='%d/%b/%Y:%H:%M:%S %z', 
    errors='coerce', 
    utc=True
)

# 3. VERIFICATION: Ensure it is actually a datetime object
#    If the format didn't match, everything became NaT (Not a Time).
#    If pandas couldn't convert it, it might stay as 'object', which causes your error.
if df['dt_obj'].dtype == 'object':
    print("Warning: Conversion failed. Trying inference...")
    # Fallback: Let pandas guess the format
    df['dt_obj'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

# 4. Debug: Check how many failed
failed_count = df['dt_obj'].isna().sum()
if failed_count > 0:
    print(f"Note: {failed_count} rows could not be parsed into dates.")

# 5. Extract features (Handle NaT/NaN safely)
df['year'] = df['dt_obj'].dt.year.fillna(0).astype(int)
df['month'] = df['dt_obj'].dt.month.fillna(0).astype(int)
df['day'] = df['dt_obj'].dt.day.fillna(0).astype(int)
df['week'] = df['dt_obj'].dt.isocalendar().week.fillna(0).astype(int)

print("Time extraction complete.")

# ==========================================
# PART C: Request Path Features
# ==========================================
print("Extracting Path features...")
def get_root_path(full_path):
    if not isinstance(full_path, str): return "/"
    parts = full_path.split('/')
    if len(parts) > 1: return "/" + parts[1]
    return "/"

df['path_feature'] = df['path'].apply(get_root_path)

# ==========================================
# PART D: IP Geolocation
# ==========================================
print("Extracting IP features...")
HAS_GEO_DB = True
GEO_DB_PATH = 'GeoLite2-City.mmdb'

if HAS_GEO_DB:
    reader = geoip2.database.Reader(GEO_DB_PATH)
    def get_geo(ip):
        try:
            r = reader.city(ip)
            return {'city': r.city.name, 'state': r.subdivisions.most_specific.name, 'country': r.country.name}
        except:
            return {'city': None, 'state': None, 'country': None}
    geo_data = df['ip'].apply(get_geo).apply(pd.Series)
    df = pd.concat([df, geo_data], axis=1)
    reader.close()
else:
    print("Skipping Geo (No DB). Setting to Unknown.")
    df['city'] = "Unknown"
    df['state'] = "Unknown"
    df['country'] = "Unknown"

# ==========================================
# PART E: LABEL GENERATION (The New Part)
# ==========================================
print("Generating Labels (Ground Truth)...")

def get_label(row):
    # Rule 1: Status 418 ("I'm a teapot") [cite: 498]
    if row['status'] == 418:
        return 1
    
    # Check request string existence
    req = str(row['request'])
    
    # Rule 2: WordPress Probes ('wp-') [cite: 500, 510]
    if 'wp-' in req:
        return 1
        
    # Rule 3: Log Harvesting ('access.log') [cite: 504, 510]
    if 'access.log' in req:
        return 1
        
    # Otherwise Benign (0)
    return 0

df['label'] = df.apply(get_label, axis=1)

# Print stats to verify
bot_count = df['label'].sum()
print(f"Labels Generated: {bot_count} Bots found out of {len(df)} total requests.")

# ==========================================
# SAVE FINAL DATASET
# ==========================================
final_columns = [
    'ip', 'browser', 'family', 'OS', 'osFamily', 
    'year', 'month', 'week', 'day', 
    'method', 'path_feature', 'status', 
    'city', 'state', 'country', 'label'  # <--- Added label here
]

df[final_columns].to_csv("processed_web_log_features.csv", index=False)
print("Done! Features and Labels saved to 'processed_web_log_features.csv'")