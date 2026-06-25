import json
from pathlib import Path

def load_config():
    # Path(__file__) is this file. We go up three levels: config.py -> model/ -> src/ -> root/
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_path = root_dir / "config.json"
    
    with open(config_path, "r") as f:
        return json.load(f)

# Instantiate globally so other files can just: `from model.config import cfg`
cfg = load_config()