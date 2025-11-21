"""
Streamlit Cloud Entry Point
Redirects to the main app in the app/ folder
"""
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the main app
from app_new import *
