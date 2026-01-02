import json
import os
import re

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_DIR = os.path.dirname(BASE_DIR)                     
DATA_DIR = os.path.join(PROJECT_DIR, "Youtube Data")        

ORIGINAL_JSON = os.path.join(DATA_DIR, "Youtube_extracted_data.json")
APPEND_JSON   = os.path.join(DATA_DIR, "Youtube_extracted_data_append.json")

# ---------------- HELPER FUNCTIONS ----------------
def recover_json_objects(path):
    """
    Recovers all valid JSON objects from a possibly corrupted JSON array file.
    Returns list of valid objects.
    """
    if not os.path.exists(path):
        print(f"❌ File does not exist: {path}")
        return []

    recovered = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Fix broken array brackets
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"

    # Remove trailing commas before ]
    text = re.sub(r",\s*]", "]", text)

    # Extract JSON objects using regex
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    print(f"Found {len(matches)} JSON-like objects in {path}.")

    for m in matches:
        try:
            obj = json.loads(m)
            recovered.append(obj)
        except json.JSONDecodeError:
            continue  # skip invalid objects

    print(f"✅ Successfully recovered {len(recovered)} valid objects.")
    return recovered

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🔹 Starting merge process...\n")

    # 1️⃣ Clean original JSON
    original_comments = recover_json_objects(ORIGINAL_JSON)
    print(f"Original JSON has {len(original_comments)} valid comments after cleanup.\n")

    # 2️⃣ Load append JSON
    append_comments = recover_json_objects(APPEND_JSON)
    print(f"Append JSON has {len(append_comments)} comments.\n")

    if not append_comments:
        print("ℹ Append JSON is empty. Nothing to merge. Exiting.")
        exit()

    # 3️⃣ Merge avoiding duplicates
    existing_keys = set(
        (c.get("videoID"), c.get("commentDate"), c.get("channelID"))
        for c in original_comments
    )
    new_comments = []
    skipped_count = 0
    for c in append_comments:
        key = (c.get("videoID"), c.get("commentDate"), c.get("channelID"))
        if key not in existing_keys:
            new_comments.append(c)
        else:
            skipped_count += 1

    print(f"✅ {len(new_comments)} new comments to merge.")
    print(f"ℹ Skipped {skipped_count} duplicate comments.\n")

    if not new_comments:
        print("ℹ No new comments to merge (all duplicates). Exiting.")
        exit()

    # 4️⃣ Merge and save back to original JSON
    merged_comments = original_comments + new_comments
    with open(ORIGINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(merged_comments, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(new_comments)} comments into {ORIGINAL_JSON}.")
    print(f"Original JSON now contains {len(merged_comments)} comments.\n")

    # 5️⃣ Clear append JSON
    with open(APPEND_JSON, "w", encoding="utf-8") as f:
        json.dump([], f)
    print(f"✅ Cleared append JSON: {APPEND_JSON}\n")
    print("🔹 Merge process complete.")
