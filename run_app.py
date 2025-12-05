#!/usr/bin/env python3
"""
Run Script for YouTube Script Generator
Quick launcher for the Streamlit application
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if all requirements are met"""
    print("Checking environment...")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if .env file exists or API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY not found!")
        print("Please create a .env file in the project root with:")
        print("   GEMINI_API_KEY=your_actual_api_key_here")
        return False
    
    # Check if data directory exists
    data_dir = Path("Data/processed")
    if not data_dir.exists():
        print("ERROR: Data/processed directory not found!")
        print("Please ensure your transcript files are in Data/processed/")
        return False
    
    # Check for transcript files
    transcript_files = list(data_dir.glob("*_transcript.json"))
    if not transcript_files:
        print("ERROR: No transcript files found!")
        print("Please ensure you have *_transcript.json files in Data/processed/")
        return False
    
    print(f"SUCCESS: Found {len(transcript_files)} transcript files")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required!")
        return False
    
    print("SUCCESS: Python version OK")
    
    return True

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("SUCCESS: Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install requirements")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("Starting YouTube Script Generator...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "light"
        ])
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"ERROR: Error running application: {e}")

def main():
    """Main function"""
    # Set UTF-8 encoding for Windows
    import sys
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    print("=" * 50)
    print("YouTube Script Generator Launcher")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\nPlease fix the issues above and try again.")
        sys.exit(1)
    
    # Ask if user wants to install requirements
    response = input("\nInstall/update requirements? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        if not install_requirements():
            sys.exit(1)
    
    print("\nStarting the application...")
    print("Open your browser to http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Run the app
    run_streamlit()

if __name__ == "__main__":
    main()

