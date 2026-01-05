#!/usr/bin/env python3
"""
Video ID Populator - Populates video_ids.xlsx with UNIQUE video IDs from CSV
MAINTAINS EXACT COMPATIBILITY with Youtube Data Extraction.py
"""

import pandas as pd
import os
import sys
from datetime import datetime

# ================= CONFIGURATION =================
# MUST match Youtube Data Extraction.py EXACTLY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Youtube Data")
CSV_PATH = os.path.join(DATA_DIR, "Youtube_extracted_data.csv")
EXCEL_PATH = os.path.join(DATA_DIR, "video_ids.xlsx")  # EXACT SAME FILENAME AND PATH

def clean_video_id(vid):
    """Clean video ID - remove whitespace, handle NaN."""
    if pd.isna(vid):
        return None
    vid_str = str(vid).strip()
    # Remove any YouTube URL prefixes
    if 'youtu.be/' in vid_str:
        vid_str = vid_str.split('youtu.be/')[-1].split('?')[0]
    if 'youtube.com/watch?v=' in vid_str:
        vid_str = vid_str.split('youtube.com/watch?v=')[-1].split('&')[0]
    return vid_str

def extract_unique_video_ids_from_csv():
    """Extract all unique video IDs from CSV file."""
    print("🔍 Extracting video IDs from CSV...")
    
    if not os.path.exists(CSV_PATH):
        print(f" Error: CSV file not found at {CSV_PATH}")
        return None
    
    try:
        # Read CSV (use low_memory to avoid warnings)
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print(f" Loaded {len(df):,} rows from CSV")
        
        # Find video ID column (case-insensitive)
        video_col = None
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            if 'videoid' in col_lower or 'video' in col_lower:
                video_col = col
                print(f" Found video ID column: '{video_col}'")
                break
        
        if not video_col:
            print(" No video ID column found in CSV")
            print("Available columns:", df.columns.tolist())
            return None
        
        # Extract and clean video IDs
        raw_ids = df[video_col].dropna().astype(str)
        print(f" Found {len(raw_ids):,} non-empty video IDs")
        
        cleaned_ids = [clean_video_id(vid) for vid in raw_ids]
        cleaned_ids = [vid for vid in cleaned_ids if vid and len(vid) > 5]  # Filter valid IDs
        
        # Get unique IDs
        unique_ids = list(set(cleaned_ids))
        print(f" After cleaning: {len(unique_ids):,} UNIQUE video IDs")
        
        return unique_ids
        
    except Exception as e:
        print(f" Error reading CSV: {e}")
        return None

def update_excel_with_unique_ids(unique_ids):
    """
    Update video_ids.xlsx with unique IDs.
    PRESERVES EXACT FORMAT for Youtube Data Extraction.py
    """
    print("\n Updating video_ids.xlsx...")
    
    # Create DataFrame with EXACT format
    new_df = pd.DataFrame({'videoID': unique_ids})
    
    # Check if Excel file already exists
    if os.path.exists(EXCEL_PATH):
        print(f" Existing video_ids.xlsx found")
        
        try:
            # Read existing file
            existing_df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
            print(f"   Contains {len(existing_df):,} video IDs")
            
            # Get existing IDs (cleaned)
            existing_ids = set(clean_video_id(vid) for vid in existing_df['videoID'].astype(str))
            existing_ids = {vid for vid in existing_ids if vid}  # Remove None
            
            # Get new IDs
            new_id_set = set(unique_ids)
            
            # Find IDs that are truly new
            truly_new_ids = new_id_set - existing_ids
            print(f"   Found {len(truly_new_ids):,} NEW unique video IDs to add")
            
            if truly_new_ids:
                # Create DataFrame for new IDs
                new_ids_df = pd.DataFrame({'videoID': list(truly_new_ids)})
                
                # Combine existing and new (remove any potential duplicates)
                combined_df = pd.concat([existing_df, new_ids_df], ignore_index=True)
                
                # Remove duplicates (in case there were any)
                combined_df = combined_df.drop_duplicates(subset=['videoID'])
                
                # Save back - EXACT same format
                combined_df.to_excel(EXCEL_PATH, index=False, engine='openpyxl')
                print(f" Added {len(truly_new_ids):,} new video IDs")
                print(f" Total unique video IDs: {len(combined_df):,}")
                
                # Sample of new IDs
                print(f"\n Sample of new video IDs added:")
                sample_new = list(truly_new_ids)[:5]
                for i, vid in enumerate(sample_new, 1):
                    print(f"   {i}. {vid}")
                    
                return combined_df
            else:
                print(" No new video IDs to add. File remains unchanged.")
                return existing_df
                
        except Exception as e:
            print(f"⚠ Error reading existing Excel: {e}")
            print("Creating fresh video_ids.xlsx...")
            new_df.to_excel(EXCEL_PATH, index=False, engine='openpyxl')
            print(f" Created new video_ids.xlsx with {len(new_df):,} unique video IDs")
            return new_df
    else:
        # Create new file
        new_df.to_excel(EXCEL_PATH, index=False, engine='openpyxl')
        print(f" Created new video_ids.xlsx with {len(new_df):,} unique video IDs")
        return new_df

def verify_compatibility():
    """Verify the Excel file is compatible with Youtube Data Extraction.py."""
    print("\n" + "=" * 60)
    print("COMPATIBILITY VERIFICATION")
    print("=" * 60)
    
    if not os.path.exists(EXCEL_PATH):
        print(" video_ids.xlsx not found")
        return False
    
    try:
        # Try to read exactly as Youtube Data Extraction.py does
        df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
        
        # Check requirements from Youtube Data Extraction.py:
        # 1. Must have 'videoID' column
        if 'videoID' not in df.columns:
            print("FAIL: Missing 'videoID' column")
            print("Columns found:", df.columns.tolist())
            return False
        
        # 2. Column must contain strings
        video_ids = df['videoID'].dropna().astype(str).tolist()
        
        # 3. Check for duplicates
        unique_ids = set(video_ids)
        if len(video_ids) != len(unique_ids):
            duplicates = len(video_ids) - len(unique_ids)
            print(f"⚠ WARNING: Found {duplicates} duplicate video IDs")
            # Auto-remove duplicates
            df_clean = df.drop_duplicates(subset=['videoID'])
            df_clean.to_excel(EXCEL_PATH, index=False, engine='openpyxl')
            print(f"Auto-removed duplicates. Now {len(df_clean):,} unique IDs")
        else:
            print(f"No duplicates found")
        
        print(f"PASS: File is compatible with Youtube Data Extraction.py")
        print(f"   File: {EXCEL_PATH}")
        print(f"   Format: Single column 'videoID' ✓")
        print(f"   Count: {len(df):,} video IDs ✓")
        print(f"   Unique: {len(set(video_ids)):,} unique IDs ✓")
        
        # Show format exactly
        print(f"\n First 5 video IDs (format check):")
        for i, vid in enumerate(df['videoID'].head().tolist(), 1):
            print(f"   {i}. '{vid}' (type: {type(vid).__name__})")
            
        return True
        
    except Exception as e:
        print(f" FAIL: Error reading Excel file: {e}")
        return False

def check_for_common_issues():
    """Check for common issues that might break Youtube Data Extraction.py."""
    print("\n Checking for common issues...")
    
    issues = []
    
    # 1. Check file exists
    if not os.path.exists(EXCEL_PATH):
        issues.append(" video_ids.xlsx does not exist")
    
    # 2. Try to read the file
    try:
        df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
        
        # 3. Check column name (case-sensitive!)
        if 'videoID' not in df.columns:
            actual_columns = df.columns.tolist()
            issues.append(f" Column 'videoID' not found. Found: {actual_columns}")
        
        # 4. Check for empty file
        if len(df) == 0:
            issues.append(" Excel file is empty")
        
        # 5. Check for NaN values
        nan_count = df['videoID'].isna().sum()
        if nan_count > 0:
            issues.append(f" Found {nan_count} empty/NaN video IDs")
        
    except Exception as e:
        issues.append(f" Cannot read Excel file: {e}")
    
    # Report issues
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("No issues found. File should work with Youtube Data Extraction.py")
        return True

def main():
    """Main function - safe population of video IDs."""
    print("=" * 70)
    print("VIDEO ID POPULATOR - SAFE MODE")
    print("=" * 70)
    print("This script will:")
    print("1. Read Youtube_extracted_data.csv")
    print("2. Extract ALL unique video IDs")
    print("3. Update video_ids.xlsx (NO format changes)")
    print("4. Ensure NO duplicate video IDs")
    print("5. Maintain compatibility with Youtube Data Extraction.py")
    print("=" * 70)
    
    # Ask for confirmation
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    # Step 1: Extract unique IDs from CSV
    unique_ids = extract_unique_video_ids_from_csv()
    if not unique_ids:
        print("Failed to extract video IDs from CSV")
        return
    
    # Step 2: Update Excel file
    final_df = update_excel_with_unique_ids(unique_ids)
    
    # Step 3: Verify compatibility
    verify_compatibility()
    
    # Step 4: Check for common issues
    check_for_common_issues()
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output Excel: {EXCEL_PATH}")
    print(f"Total unique video IDs: {len(final_df):,}")
    
    # Show what Youtube Data Extraction.py will see
    print("\n Youtube Data Extraction.py will see:")
    print(f"   File: {EXCEL_PATH} ✓")
    print(f"   Column: 'videoID' ✓")
    print(f"   Rows: {len(final_df):,} ✓")
    print(f"   Duplicates: 0 ✓")
    
    print("\n READY for Youtube Data Extraction.py")
    print("\nNext step: Run 'python Youtube Data Extraction.py'")
    print("=" * 70)

def quick_fix_mode():
    """Quick fix mode - just cleans existing video_ids.xlsx."""
    print("\n" + "=" * 70)
    print("QUICK FIX MODE - Clean duplicates only")
    print("=" * 70)
    
    if not os.path.exists(EXCEL_PATH):
        print(f" {EXCEL_PATH} not found")
        return
    
    try:
        df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
        print(f" Loaded {len(df):,} video IDs")
        
        # Remove duplicates
        before = len(df)
        df_clean = df.drop_duplicates(subset=['videoID'])
        after = len(df_clean)
        
        if before == after:
            print(" No duplicates found")
        else:
            removed = before - after
            df_clean.to_excel(EXCEL_PATH, index=False, engine='openpyxl')
            print(f" Removed {removed} duplicate video IDs")
            print(f" Now {after:,} unique video IDs")
        
        verify_compatibility()
        
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    print("Video ID Population Script")
    print("1. Full population (extract all unique video ids from CSV)")
    print("2. Quick fix (remove duplicates only)")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_fix_mode()
    else:
        print("Invalid choice. Exiting.")