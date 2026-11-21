"""Build the local model artifact from the bundled weekly CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dengue.model import MODEL_PATH, train_model_bundle  # noqa: E402


if __name__ == "__main__":
    bundle = train_model_bundle(output_path=MODEL_PATH)
    print(
        json.dumps(
            {
                "model_path": str(MODEL_PATH.relative_to(ROOT)),
                "data_summary": bundle["data_summary"],
                "metrics": bundle["metrics"],
                "versions": bundle["versions"],
            },
            indent=2,
        )
    )
