from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from main import app  # noqa: E402


def main():
    return app


if __name__ == "__main__":
    main()
