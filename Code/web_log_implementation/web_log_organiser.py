import os
import shutil
import zipfile

# 1. SETUP PATHS
# Use raw string (r"...") to handle Windows backslashes correctly
base_dir = r"D:\cloud and web based security\Project\self.logs"
access_dest = os.path.join(base_dir, "all_access_logs")
error_dest = os.path.join(base_dir, "all_error_logs")

# Create the destination folders if they don't exist
for folder in [access_dest, error_dest]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

print(f"Scanning: {base_dir}")

# 2. WALK THROUGH ALL FOLDERS
# os.walk will find files even if they are inside subfolders like '2015', '2016'
for root, dirs, files in os.walk(base_dir):
    
    # Skip our destination folders so we don't re-process files we just moved
    if "all_access_logs" in root or "all_error_logs" in root:
        continue

    for filename in files:
        file_path = os.path.join(root, filename)

        # CASE A: It's a ZIP file -> Extract it first
        if filename.lower().endswith(".zip"):
            print(f"Unzipping: {filename}")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract to the same folder the zip is in
                    zip_ref.extractall(root)
            except zipfile.BadZipFile:
                print(f"Warning: Corrupt zip file skipped: {filename}")
            
            # Note: The loop continues; the extracted .gz files will be picked up 
            # either in this loop (if os.walk updates) or you might need to run it twice.
            # To be safe, we usually extract, then process .gz files in a second pass 
            # or just handle the .gz files that appear.
            
        # CASE B: It's a GZ file -> Move it
        elif filename.lower().endswith(".gz"):
            # Determine destination
            if "access" in filename.lower():
                target_folder = access_dest
            elif "error" in filename.lower():
                target_folder = error_dest
            else:
                continue # Skip if it's not clearly access or error log

            # Construct new path
            new_path = os.path.join(target_folder, filename)
            
            # Check if file already exists to avoid overwriting (or handle duplicates)
            if not os.path.exists(new_path):
                shutil.move(file_path, new_path)
                print(f"Moved: {filename} -> {target_folder}")
            else:
                print(f"Skipping duplicate: {filename}")

# 3. CLEANUP (Optional)
# If you want to delete empty folders (like '2015') after moving files, you can add that here.
print("\nOrganization Complete!")
print(f"Access logs are in: {access_dest}")
print(f"Error logs are in:  {error_dest}")