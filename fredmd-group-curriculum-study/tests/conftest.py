from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
sys.path.insert(0, str(ROOT / "src"))
