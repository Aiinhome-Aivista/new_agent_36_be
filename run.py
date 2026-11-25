"""
Entry point for the DevKnow API Server.
Initializes the Flask application, loads configuration, and binds to the specified host/port.
Usage:
    python run.py
"""
import os
import warnings

# Suppress legacy package deprecation notice (e.g. tree-sitter C-extension warnings) in console
warnings.filterwarnings("ignore", category=FutureWarning)

from app import create_app
from app.config import Config

# Create Flask application instance via application factory pattern
app = create_app()

if __name__ == "__main__":
    # Retrieve runtime host and port from environment or default configuration
    port = int(os.getenv("API_PORT", Config.API_PORT))
    host = os.getenv("API_HOST", Config.API_HOST)
    
    print(f"\n=======================================================")
    print(f"🚀 DevKnow API Server starting at http://{host}:{port}")
    print(f"📡 API Documentation / Base Prefix: {Config.API_PREFIX}")
    print(f"=======================================================\n")
    
    # Run development server with live reload enabled if Config.DEBUG is True
    app.run(host=host, port=port, debug=Config.DEBUG)


