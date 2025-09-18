#!/usr/bin/env python3
"""
Workflow Script for Inference Testing
====================================

This script runs all the inference tests in sequence:
1. Dummy data testing
2. Uncertainty demonstration
3. Albuquerque real data testing

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """
    Run a Python script and capture its output.
    
    Parameters:
    -----------
    script_name : str
        Name of the script to run
    description : str
        Description of what the script does
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"✗ Script {script_name} not found!")
        return False
    
    try:
        # Run the script
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check result
        if result.returncode == 0:
            print(f"✓ {description} completed successfully in {end_time - start_time:.1f} seconds")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Error running {description}: {str(e)}")
        return False

def main():
    """
    Main workflow function.
    """
    print("=" * 60)
    print("INFERENCE WORKFLOW - URBAN INFORMATION PROJECT")
    print("=" * 60)
    
    print("\nThis workflow will run all inference tests in sequence:")
    print("1. Dummy data testing (baseline validation)")
    print("2. Uncertainty demonstration (synthetic data)")
    print("3. Albuquerque real data testing")
    
    # Check current directory
    if not os.path.exists("vinference.py"):
        print("\n✗ Please run this script from the inference directory!")
        print("Current directory:", os.getcwd())
        return
    
    print(f"\n✓ Running from inference directory: {os.getcwd()}")
    
    # Step 1: Test with dummy data
    success1 = run_script("test_vinference.py", "Dummy Data Testing")
    
    # Step 2: Run uncertainty demonstration
    success2 = run_script("demo_uncertainty.py", "Uncertainty Demonstration")
    
    # Step 3: Test with Albuquerque data
    if os.path.exists("albuquerque_extracted_data.pkl"):
        success3 = run_script("test_albuquerque_vinference.py", "Albuquerque Real Data Testing")
    else:
        print("\n⚠ Albuquerque extracted data not found.")
        print("To test with real data, first run:")
        print("  python examine_albuquerque_data.py")
        success3 = False
    
    # Summary
    print(f"\n{'='*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*60}")
    
    tests = [
        ("Dummy Data Testing", success1),
        ("Uncertainty Demonstration", success2),
        ("Albuquerque Real Data Testing", success3)
    ]
    
    for test_name, success in tests:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Examine the generated results files")
        print("2. Review the visualization plots")
        print("3. Analyze the differences between methods")
        print("4. Implement improvements from upgrades.txt")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that data files are present")
        print("3. Verify script permissions and paths")

if __name__ == "__main__":
    main() 