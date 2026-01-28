import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent  
ROOT_DIR = BASE_DIR.parent 

DATA_PATH = ROOT_DIR / "player-performance" / "data" / "processed" / "players_full_2425_with_score.csv"

ARTIFACTS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR.mkdir(exist_ok=True)

SCALER_PATH = ARTIFACTS_DIR / "similarity_scaler.pkl"
PCA_PATH = ARTIFACTS_DIR / "similarity_pca.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "similarity_features.pkl"
META_PATH = ARTIFACTS_DIR / "similarity_metadata.pkl"


SIM_FEATURES = [
    "Age", "90s",
    "Gls_90", "Ast_90", "G+A_90",
    "G-PK_90", "G+A-PK_90",
    "xG_90", "xAG_90", "xG+xAG_90",
    "npxG_90", "npxG+xAG_90",
    "PrgC", "PrgP", "PrgR"
]


def train_and_save(min_90s=5, n_components=8):
    df = pd.read_csv(DATA_PATH)
    df = df[df["90s"] >= min_90s].copy()

    df_sim = df.dropna(subset=SIM_FEATURES).copy()
    meta = df_sim[["Player", "Squad", "Pos"]].copy()

    X = df_sim[SIM_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)
    joblib.dump(SIM_FEATURES, FEATURES_PATH)
    joblib.dump(meta, META_PATH)

    print("Saved similarity artifacts to:", ARTIFACTS_DIR)
    print("Players used:", len(df_sim))
    print("Explained variance:", float(pca.explained_variance_ratio_.sum()))


def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    features = joblib.load(FEATURES_PATH)
    meta = joblib.load(META_PATH)
    return scaler, pca, features, meta


def build_embeddings(min_90s=5):
    df = pd.read_csv(DATA_PATH)
    df = df[df["90s"] >= min_90s].copy()
    df_sim = df.dropna(subset=SIM_FEATURES).copy()

    scaler, pca, features, meta = load_artifacts()

    X = df_sim[features].values
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    return df_sim, X_pca


def find_similar(player_name, top_n=10, same_position=True, min_90s=5):
    df_sim, X_pca = build_embeddings(min_90s=min_90s)

    # find player
    exact = df_sim["Player"].astype(str).str.lower() == player_name.lower()
    if not exact.any():
        contains = df_sim["Player"].astype(str).str.lower().str.contains(player_name.lower(), na=False)
        matches = df_sim[contains][["Player", "Squad", "Pos"]].head(10)
        return None, "Player not found. Closest matches:\n" + matches.to_string(index=False)

    idx_label = df_sim.index[exact][0]
    idx = df_sim.index.get_loc(idx_label)

    player_pos = df_sim.iloc[idx]["Pos"]

    sims = cosine_similarity(X_pca[idx].reshape(1, -1), X_pca).flatten()
    ranked = np.argsort(sims)[::-1]

    out = []
    for j in ranked:
        if j == idx:
            continue
        if same_position and df_sim.iloc[j]["Pos"] != player_pos:
            continue

        out.append([
            df_sim.iloc[j]["Player"],
            df_sim.iloc[j]["Squad"],
            df_sim.iloc[j]["Pos"],
            float(sims[j])
        ])
        if len(out) >= top_n:
            break

    return pd.DataFrame(out, columns=["Player", "Squad", "Pos", "Similarity"]), None


if __name__ == "__main__":
    if not (SCALER_PATH.exists() and PCA_PATH.exists() and FEATURES_PATH.exists() and META_PATH.exists()):
        train_and_save()

    name = input("Enter player name: ").strip()
    res, err = find_similar(name, top_n=10, same_position=True)

    if err:
        print(err)
    else:
        print(res.to_string(index=False))
