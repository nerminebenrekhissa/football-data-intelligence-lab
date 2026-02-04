from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd


@dataclass(frozen=True)
class PlayerCandidate:
    idx: int
    Player: str
    Squad: str
    Comp: str
    Pos: str
    Age: Optional[float] = None


def _norm(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def build_player_uid(row: pd.Series) -> str:
    return f"{row.get('Player','')}_{row.get('Squad','')}_{row.get('Comp','')}_{row.get('Pos','')}".strip()


def resolve_player(
    df: pd.DataFrame,
    player: str,
    squad: Optional[str] = None,
    comp: Optional[str] = None,
    pos: Optional[str] = None,
    allow_partial: bool = True,
) -> Tuple[pd.Series, List[PlayerCandidate]]:
    """
    Resolve a player safely without silent guessing.

    Returns:
      - row (pd.Series) + empty list if uniquely resolved
      - empty row + candidate list if ambiguous or not found
    """
    if "Player" not in df.columns:
        raise ValueError("DataFrame must contain a 'Player' column.")

    p = _norm(player)
    name_series = df["Player"].astype(str)

    exact_mask = name_series.str.strip().str.lower().eq(p)
    matches = df[exact_mask].copy()

    if matches.empty and allow_partial:
        contains_mask = name_series.str.strip().str.lower().str.contains(p, na=False)
        matches = df[contains_mask].copy()

    if squad and "Squad" in matches.columns:
        s = _norm(squad)
        matches = matches[matches["Squad"].astype(str).str.strip().str.lower().eq(s)]

    if comp and "Comp" in matches.columns:
        c = _norm(comp)
        matches = matches[matches["Comp"].astype(str).str.strip().str.lower().eq(c)]

    if pos and "Pos" in matches.columns:
        po = _norm(pos)
        matches = matches[matches["Pos"].astype(str).str.strip().str.lower().eq(po)]

    if len(matches) == 1:
        row = matches.iloc[0].copy()
        row["player_uid"] = build_player_uid(row)
        return row, []

    # build candidates list 
    candidates: List[PlayerCandidate] = []
    for _, r in matches.reset_index(drop=False).iterrows():
        candidates.append(
            PlayerCandidate(
                idx=int(r["index"]),
                Player=str(r.get("Player", "")),
                Squad=str(r.get("Squad", "")),
                Comp=str(r.get("Comp", "")),
                Pos=str(r.get("Pos", "")),
                Age=r.get("Age", None),
            )
        )

    return pd.Series(dtype=object), candidates


def pick_candidate_interactive(candidates: List[PlayerCandidate]) -> Optional[int]:
    """
    CLI selection helper.
    Returns chosen df index, or None if cancelled.
    """
    if not candidates:
        return None

    print("\nMultiple players match your query. Please choose one:")
    for k, c in enumerate(candidates, start=1):
        age_str = "" if c.Age is None else f", Age={c.Age}"
        print(f"{k}) {c.Player} | {c.Squad} | {c.Comp} | {c.Pos}{age_str}")

    while True:
        choice = input("Enter number (or press Enter to cancel): ").strip()
        if choice == "":
            return None
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(candidates):
                return candidates[n - 1].idx
        print("Invalid choice. Try again.")
