#!/usr/bin/env python3
"""
Test script to verify data_prep.py can find all required components.
"""

import os
from pathlib import Path

def test_file_locations():
    """Test that all required files and scripts exist."""
    base_dir = Path(__file__).parent
    
    print("🔍 Testing Data Preparation Pipeline Components")
    print("=" * 50)
    
    # Test ACS data retrieval
    acs_script = base_dir / "data_retrieval" / "acs_data_retrieval.py"
    print(f"ACS Script: {'✅' if acs_script.exists() else '❌'} {acs_script}")
    
    # Test ZIP matching
    zip_script = base_dir / "zip_matching" / "match_zips.py"
    print(f"ZIP Script: {'✅' if zip_script.exists() else '❌'} {zip_script}")
    
    # Test data processing
    process_script = base_dir / "process_acs_data.py"
    print(f"Process Script: {'✅' if process_script.exists() else '❌'} {process_script}")
    
    # Test data_prep.py itself
    data_prep_script = base_dir / "data_prep.py"
    print(f"Data Prep Script: {'✅' if data_prep_script.exists() else '❌'} {data_prep_script}")
    
    print("\n📁 Checking Required Data Files:")
    
    # Check if ZIP data exists
    uszips_file = base_dir / "zip_matching" / "uszips.csv"
    print(f"ZIP Database: {'✅' if uszips_file.exists() else '❌'} {uszips_file}")
    
    # Check if centroid data exists
    centroids_2010 = base_dir / "zip_matching" / "nhgis0001_shapefile_cenpop2010_us_blck_grp_cenpop_2010"
    centroids_2020 = base_dir / "zip_matching" / "nhgis0001_shapefile_cenpop2020_us_blck_grp_cenpop_2020"
    
    print(f"2010 Centroids: {'✅' if centroids_2010.exists() else '❌'} {centroids_2010}")
    print(f"2020 Centroids: {'✅' if centroids_2020.exists() else '❌'} {centroids_2020}")
    
    # Check if existing data files exist
    acs_data = base_dir / "data_retrieval" / "cbsa_acs_data.pkl"
    zip_matched = base_dir / "zip_matching" / "blockgroups_with_zips_temporal.pkl"
    processed = base_dir / "data_retrieval" / "processed_acs_data.pkl"
    
    print(f"\n📊 Existing Data Files:")
    print(f"Raw ACS Data: {'✅' if acs_data.exists() else '❌'} {acs_data}")
    print(f"ZIP-Matched Data: {'✅' if zip_matched.exists() else '❌'} {zip_matched}")
    print(f"Processed Data: {'✅' if processed.exists() else '❌'} {processed}")
    
    # Summary
    print("\n" + "=" * 50)
    all_scripts_exist = all([
        acs_script.exists(),
        zip_script.exists(),
        process_script.exists(),
        data_prep_script.exists()
    ])
    
    if all_scripts_exist:
        print("🎉 All required scripts found! Data preparation pipeline should work.")
        print("\nTo run the pipeline:")
        print(f"cd {base_dir}")
        print("python data_prep.py")
    else:
        print("❌ Some required scripts are missing. Please check the file paths above.")
    
    return all_scripts_exist

if __name__ == "__main__":
    test_file_locations() 