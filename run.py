import os
from app import create_app
from app.config import Config

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", Config.API_PORT))
    host = os.getenv("API_HOST", Config.API_HOST)
    print(f"\n=======================================================")
    print(f"🚀 DevKnow API Server starting at http://{host}:{port}")
    print(f"📡 API Documentation / Base Prefix: {Config.API_PREFIX}")
    print(f"=======================================================\n")
    app.run(host=host, port=port, debug=Config.DEBUG)

