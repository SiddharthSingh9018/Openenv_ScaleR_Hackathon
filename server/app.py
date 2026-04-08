from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
TARGET_APP_DIR = ROOT / "pedestrian-negotiation-env" / "app"
sys.path.insert(0, str(TARGET_APP_DIR))

from main import app  # noqa: E402


def main():
    return app


if __name__ == "__main__":
    main()
