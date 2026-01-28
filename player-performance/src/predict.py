import json
import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  

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

    mask = df["Player"].astype(str).str.lower().str.contains(player_name.lower(), na=False)
    matches = df[mask].copy()

    if matches.empty:
        print(f"No player found containing: {player_name}")
        return

    row = matches.iloc[0]

    X = pd.DataFrame([row[features].values], columns=features)

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    print("Player:", row["Player"])
    print("Team:", row.get("Squad", "N/A"))
    print("Position:", row.get("Pos", "N/A"))
    print(f"Predicted good performance (proxy): {pred}")
    print(f"Probability (class 1): {proba:.3f}")

    if len(matches) > 1:
        print("\nOther matches found:")
        print(matches[["Player", "Squad", "Pos"]].head(top_k).to_string(index=False))

if __name__ == "__main__":
    name = input("Enter player name: ").strip()
    predict_player(name)
