"""Central configuration: paths, label order, and feature vocabulary.

The feature aliases are the bridge between the ML world (column names) and the
natural-language world (what an LLM narrative actually writes). They are used by
`faithfulness.py` to detect which features a generated explanation refers to.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv():
    """Minimal dependency-free .env loader: KEY=VALUE lines into the environment
    (never overriding variables already set in the shell)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()
DATA = ROOT / "data" / "online_gaming_behavior_dataset.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
EXPLANATIONS = RESULTS / "explanations"
for _p in (FIGURES, MODELS, EXPLANATIONS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "EngagementLevel"
# Ordered from least to most engaged so class indices are meaningful.
CLASSES = ["Low", "Medium", "High"]

# LLM defaults (overridable via env in llm.py). llama3.2 is small + fast for a
# live demo; llama3 (8B) gives richer prose if time allows.
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"
OPENAI_MODEL = "gpt-4o-mini"

# Canonical feature -> phrases a human/LLM might use for it. Longest phrases are
# matched first so "session duration" wins over "session". The raw column token
# (e.g. "sessionsperweek") is included so narratives that quote column names are
# also detected. Note: bare "level" is deliberately excluded from PlayerLevel
# because "engagement level" appears constantly and would cause false matches.
FEATURE_ALIASES = {
    "SessionsPerWeek": ["sessions per week", "weekly sessions", "session frequency",
                        "number of sessions", "play frequency", "how often",
                        "sessionsperweek", "sessions"],
    "AvgSessionDurationMinutes": ["average session duration", "session duration",
                                  "session length", "minutes per session",
                                  "time per session", "length of their sessions",
                                  "avgsessiondurationminutes", "how long"],
    "PlayTimeHours": ["play time", "playtime", "hours played", "total hours",
                      "hours of play", "playing hours", "playtimehours"],
    "AchievementsUnlocked": ["achievements unlocked", "achievementsunlocked",
                             "achievements", "unlocked"],
    "PlayerLevel": ["player level", "player's level", "character level", "playerlevel"],
    "Age": ["age"],
    "InGamePurchases": ["in-game purchases", "in game purchases", "ingamepurchases",
                        "purchases", "spending", "spend"],
    "GameGenre": ["genre", "strategy", "sports", "action", "rpg", "simulation"],
    "GameDifficulty": ["difficulty", "easy difficulty", "hard difficulty"],
    "Gender": ["gender", "male", "female"],
    "Location": ["location", "region", "usa", "europe", "asia"],
}

# Human-readable labels for narratives (replaces raw CamelCase column names).
BASE_LABELS = {
    "SessionsPerWeek": "sessions per week",
    "AvgSessionDurationMinutes": "average session duration",
    "PlayTimeHours": "play time (hours)",
    "AchievementsUnlocked": "achievements unlocked",
    "PlayerLevel": "player level",
    "Age": "age",
    "InGamePurchases": "in-game purchases",
}
_DUMMY_LABELS = {"GameGenre": "{} genre", "Gender": "{}", "Location": "{}",
                 "GameDifficulty": "{} difficulty"}


def readable(column: str) -> str:
    """Turn a model column into narrative-friendly text.
    'GameGenre_RPG' -> 'RPG genre', 'SessionsPerWeek' -> 'sessions per week'."""
    if column in BASE_LABELS:
        return BASE_LABELS[column]
    if "_" in column:
        base, val = column.split("_", 1)
        if base in _DUMMY_LABELS:
            return _DUMMY_LABELS[base].format(val)
    return column

# Words signalling the magnitude a sentence attributes to a feature value.
POLARITY_HIGH = ["high", "higher", "more", "many", "frequent", "frequently", "long",
                 "longer", "increase", "increases", "greater", "large", "strong",
                 "elevated", "above"]
POLARITY_LOW = ["low", "lower", "few", "fewer", "less", "rare", "rarely", "short",
                "shorter", "decrease", "decreases", "smaller", "weak", "below",
                "limited", "minimal"]
