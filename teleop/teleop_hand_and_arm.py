"""Compatibility launcher. Prefer ``python collect.py`` from the repository root."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    collect = Path(__file__).resolve().parents[1] / "collect.py"
    runpy.run_path(str(collect), run_name="__main__")
