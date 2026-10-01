"""FastAPI backend for the loan default model.

The ML code lives in src/ (it uses plain imports like `from load_data import ...`), so src/ is put on the
import path once here - every `import app.<module>` runs this file first.
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
