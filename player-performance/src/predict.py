import json
import joblib
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent

COMMON_SRC = ROOT_DIR / "common" / "src"
sys.path.append(str(COMMON_SRC))

from player_id import resolve_player, pick_candidate_interactive

MODEL_PATH = BASE_DIR / "models" / "player_performance_model_B.pkl"
FEATURES_PATH = BASE_DIR / "models" / "player_performance_features_B.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "players_full_2425_with_score.csv"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    return model, features


def predict_player(player_name: str, top_k=5):
    model, features = load_artifacts()
    df = pd.read_csv(DATA_PATH)

    row, candidates = resolve_player(df, player_name, allow_partial=True)

    if len(candidates) > 0:
        chosen_idx = pick_candidate_interactive(candidates)
        if chosen_idx is None:
            print("Cancelled.")
            return
        row = df.loc[chosen_idx].copy()

        print("\nOther matches were:")
        for c in candidates[:top_k]:
            print(f"- {c.Player} | {c.Squad} | {c.Comp} | {c.Pos}")

    if row.empty:
        print(f"No player found matching: {player_name}")
        return

    X = pd.DataFrame([row[features].values], columns=features)

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    print("\nPlayer:", row["Player"])
    print("Team:", row.get("Squad", "N/A"))
    print("Position:", row.get("Pos", "N/A"))
    print(f"Predicted good performance (proxy): {pred}")
    print(f"Probability (class 1): {proba:.3f}")


if __name__ == "__main__":
    name = input("Enter player name: ").strip()
    predict_player(name)
