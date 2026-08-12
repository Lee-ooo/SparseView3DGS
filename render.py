"""Project-root rendering entry point.

This wrapper keeps the implementation in ``program/scripts/render_views.py``
while allowing rendering to be started directly from the project root.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "program" / "scripts" / "render_views.py"),
        run_name="__main__",
    )
