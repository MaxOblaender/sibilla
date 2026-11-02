"""pytest configuration for sibilla.

With the src-layout the package lives in `src/sibilla`. Add `src` to
sys.path so `import sibilla` works from the repo root without an editable
install.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
