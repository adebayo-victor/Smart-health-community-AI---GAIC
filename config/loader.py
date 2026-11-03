import json
import os
from pathlib import Path

# Dynamically find the root directory (2 levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

# Load the JSON config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

# Extract "Slots" into clean, easy-to-import variables
PATHS = SETTINGS["paths"]
MODELS = SETTINGS["models"]
API = SETTINGS["api"]
LANGUAGES = SETTINGS["languages"]
PARAMS = SETTINGS["pipeline_params"]
QUEUES = SETTINGS["worker_queues"]
SAFETY = SETTINGS["safety_gate"]

# Ensure required directories exist on startup
os.makedirs(PATHS["audio_output_dir"], exist_ok=True)
os.makedirs(PATHS["local_models_dir"], exist_ok=True)
os.makedirs(PATHS["pdf_directory"], exist_ok=True)

print("✅ [Config Loader] Settings loaded successfully.")
