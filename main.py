"""Main entrypoint — runs the FastAPI server with uvicorn."""
import uvicorn
from src.config import Config

if __name__ == "__main__":
    cfg = Config.load()
    uvicorn.run(
        "src.api.app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=cfg.api.reload,
    )
