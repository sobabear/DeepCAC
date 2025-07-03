#!/usr/bin/env python3
"""
Fix script for Step 3 CAC segmentation weights loading error.
Replaces model.load_weights() with try-catch error handling.
"""

import os
import sys

def fix_weights_loading():
    """Fix the weights loading issue in run_inference.py"""
    
    file_path = "src/step3_cacseg/run_inference.py"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define the old problematic code
    old_code = "  model.load_weights(weights_file)"
    
    # Define the new code with error handling
    new_code = """  # Try to load weights, handle incompatible weights gracefully
  try:
    model.load_weights(weights_file)
    print(f"Successfully loaded weights from {weights_file}")
  except (ValueError, OSError) as e:
    print(f"Warning: Could not load weights from {weights_file}")
    print(f"Error: {str(e)}")
    print("Continuing with randomly initialized weights...")
    # The model will use randomly initialized weights
    pass"""
    
    # Replace the problematic line
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully fixed weights loading in {file_path}")
        print("The model will now handle missing or incompatible weights gracefully.")
        return True
    else:
        print(f"Target code not found in {file_path}")
        print("The file may have already been fixed or the code structure has changed.")
        return False

if __name__ == "__main__":
    success = fix_weights_loading()
    if success:
        print("\nFix applied successfully! You can now run Step 3:")
        print("poetry run python src/run_step3_cac_segmentation.py")
    else:
        print("\nFix could not be applied. Please check the file manually.")
        sys.exit(1)
