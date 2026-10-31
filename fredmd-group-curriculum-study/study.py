"""Command-line entry point; all generated work stays in this research folder."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(ROOT / "src"))

from fredmd_study.cli import main

if __name__ == "__main__":
    main(ROOT)
