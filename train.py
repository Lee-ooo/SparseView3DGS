"""Project-root training entry point.

This wrapper keeps the implementation in ``program/scripts/train_model.py``
while allowing training to be started directly from the project root.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "program" / "scripts" / "train_model.py"),
        run_name="__main__",
    )
