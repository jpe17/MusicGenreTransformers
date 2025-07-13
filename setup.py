#!/usr/bin/env python3
"""
Setup script for Music Genre Classification with Transformers
"""

import os
import sys
import subprocess

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("🔧 Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print("✅ Virtual environment created successfully!")

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Determine the activation script path
    if sys.platform == "win32":
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:
        pip_path = os.path.join("venv", "bin", "pip")
    
    subprocess.run([pip_path, "install", "-r", "backend/requirements.txt"], check=True)
    print("✅ Dependencies installed successfully!")

def check_model_exists():
    """Check if the trained model exists"""
    model_path = "backend/music_genre_classifier.pth"
    if os.path.exists(model_path):
        print(f"✅ Trained model found at {model_path}")
        return True
    else:
        print(f"⚠️  Trained model not found at {model_path}")
        print("   You'll need to train the model first by running:")
        print("   cd backend && python train.py")
        return False

def main():
    print("🎵 Music Genre Classification with Transformers - Setup")
    print("=" * 55)
    
    try:
        # Create virtual environment
        create_virtual_environment()
        
        # Install dependencies
        install_dependencies()
        
        # Check for trained model
        model_exists = check_model_exists()
        
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if sys.platform == "win32":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        
        if not model_exists:
            print("2. Train the model:")
            print("   cd backend && python train.py")
            print("3. Run the application:")
            print("   python app.py")
        else:
            print("2. Run the application:")
            print("   cd backend && python app.py")
        
        print("\n📖 Check the README.md for more detailed instructions!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 