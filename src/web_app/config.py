import os
from pathlib import Path

class AppConfig:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "resnet3d_best_10_01_16:49.pt"
    
    MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
