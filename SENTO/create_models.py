# create_models.py
import os
import sys

# Add the current directory to path so we can import your modules
sys.path.append('.')

from model_initializer import ModelInitializer

if __name__ == "__main__":
    print("🔄 Creating missing model files...")
    initializer = ModelInitializer()
    initializer.initialize_all_models()
    print("✅ Model creation complete!")