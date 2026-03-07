"""
Compatibility entrypoint.
Primary FastAPI app now lives in backend/app.py.
"""

from backend.app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
