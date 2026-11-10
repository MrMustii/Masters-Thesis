"""Regional GenCast — model, data layer, metrics and reporting for the thesis experiments.

Puts the repository root and ``Googles_gencast`` on ``sys.path``, so modules below can
``from graphcast import ...`` without each carrying its own path setup.
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOOGLES_GENCAST = REPOSITORY_ROOT / "Googles_gencast"

for search_path in (GOOGLES_GENCAST, REPOSITORY_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
