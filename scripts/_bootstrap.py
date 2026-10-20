"""Put ../src on sys.path so scripts can `import gamexai` without installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
