# run_ngrok.py

from pyngrok import ngrok
import uvicorn

# Open ngrok tunnel to the FastAPI app
public_url = ngrok.connect(8000)
print(f" Ngrok Tunnel URL: {public_url}")

# Start the FastAPI app
uvicorn.run("ensemble_api:app", host="0.0.0.0", port=8000, reload=True)