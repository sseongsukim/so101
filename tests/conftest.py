import sys
from pathlib import Path

# Test this repo's src, not a sibling editable install of `so101`
# (see scripts/make_calibration_targets.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
