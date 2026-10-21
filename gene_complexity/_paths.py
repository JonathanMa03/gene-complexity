"""Repository and packaged resource locations (editable installation recommended)."""
import os
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GENE_COMPLEXITY_ROOT", PACKAGE_ROOT.parent)).resolve()
CONFIG = PACKAGE_ROOT / "classification/configs/validation_v1.json"
