"""
Data Collection Node - Step 2 of the Training Agent

Searches local sources and (optionally) external sources via the Dataset
Curator to find the best training dataset for the goal.
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

_DATA_RETRIEVAL_DIR = Path(__file__).parent.parent.parent / "data-retrieval"
if str(_DATA_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_RETRIEVAL_DIR))

from agent import DataRetrievalResult, retrieve_data
from utils import get_registered_dataset, register_dataset

if TYPE_CHECKING:
    from ..core.state import TrainingAgentState

_MIN_ROWS = 10
_MIN_COLUMNS = 2
_MARCH_MADNESS_DIR = Path(__file__).parent.parent.parent.parent / ".kaggle" / "march-machine-learning-mania-2026"


# =============================================================================
# Helpers
# =============================================================================


def _validate(ref: str) -> list[str]:
    """Return a list of issues (empty = valid)."""
    df = get_registered_dataset(ref)
    if df is None:
        return [f"Dataset '{ref}' not found in registry"]
    issues: list[str] = []
    if len(df) < _MIN_ROWS:
        issues.append(f"Too few rows: {len(df)} (minimum {_MIN_ROWS})")
    if len(df.columns) < _MIN_COLUMNS:
        issues.append(f"Too few columns: {len(df.columns)} (minimum {_MIN_COLUMNS})")
    if df.shape[0] > 0 and df.isna().all(axis=0).any():
        all_null = [c for c in df.columns if df[c].isna().all()]
        issues.append(f"Entirely null columns: {all_null}")
    if len(df) > 0 and len(df.drop_duplicates()) == 1:
        issues.append("All rows are identical")
    return issues


def _infer_target_column(df: pd.DataFrame, goal: str) -> str | None:
    """Best-effort guess at which column is the prediction target."""
    goal_lower = goal.lower()
    cols = list(df.columns)
    for col in cols:
        if col.lower() in goal_lower:
            return col
    target_keywords = [
        "target", "label", "class", "churn", "price", "status",
        "outcome", "approved", "default", "fraud", "condition",
        "survived", "diagnosis", "y",
    ]
    for kw in target_keywords:
        for col in cols:
            if kw in col.lower():
                return col
    return None


def _signal_strength(df: pd.DataFrame, target_col: str) -> float:
    """Return average |correlation| between numeric features and the target.

    Falls back to 0.0 if the target is non-numeric or there are no features.
    """
    if target_col not in df.columns:
        return 0.0
    target = df[target_col]
    if not np.issubdtype(target.dtype, np.number):
        try:
            target = target.astype(float)
        except (ValueError, TypeError):
            return 0.0

    numeric = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    if numeric.empty:
        return 0.0
    corrs = numeric.corrwith(target).abs().dropna()
    return float(corrs.mean()) if len(corrs) > 0 else 0.0


def _goal_relevance(df: pd.DataFrame, goal: str) -> float:
    """Score how well a dataset's columns match the goal keywords. Returns 0..1."""
    if not goal:
        return 0.0
    goal_words = set(re.sub(r"[^a-z0-9 ]", " ", goal.lower()).split())
    # Remove stop words that would match anything
    goal_words -= {
        "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is",
        "it", "by", "as", "at", "be", "if", "do", "from", "with", "that",
        "this", "will", "can", "like", "such", "based", "using", "train",
        "model", "predict", "build", "use", "data", "dataset", "learning",
        "whether", "score", "neural", "network", "deep",
    }
    if not goal_words:
        return 0.0
    col_words = set()
    for col in df.columns:
        col_words.update(re.sub(r"[^a-z0-9 ]", " ", col.lower()).split())
    matches = goal_words & col_words
    return len(matches) / len(goal_words)


def _is_march_madness_goal(goal: str) -> bool:
    text = (goal or "").lower()
    keywords = (
        "march madness",
        "machine learning mania",
        "ncaa",
        "tournament",
        "kaggle",
        "brier",
        "bracket",
        "basketball",
    )
    return any(k in text for k in keywords)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _extract_seed_number(seed_value: str) -> float:
    if not isinstance(seed_value, str) or len(seed_value) < 3:
        return np.nan
    try:
        return float(int(seed_value[1:3]))
    except Exception:
        return np.nan


def _perspective_rows_from_detailed(df: pd.DataFrame) -> pd.DataFrame:
    winner = pd.DataFrame({
        "Season": df["Season"],
        "DayNum": df["DayNum"],
        "TeamID": df["WTeamID"],
        "OppTeamID": df["LTeamID"],
        "Score": df["WScore"],
        "OppScore": df["LScore"],
        "Win": 1,
        "Loc": df["WLoc"],
        "NumOT": df["NumOT"],
        "FGM": df["WFGM"],
        "FGA": df["WFGA"],
        "FGM3": df["WFGM3"],
        "FGA3": df["WFGA3"],
        "FTM": df["WFTM"],
        "FTA": df["WFTA"],
        "OR": df["WOR"],
        "DR": df["WDR"],
        "Ast": df["WAst"],
        "TO": df["WTO"],
        "Stl": df["WStl"],
        "Blk": df["WBlk"],
        "PF": df["WPF"],
        "OppFGM": df["LFGM"],
        "OppFGA": df["LFGA"],
        "OppFGM3": df["LFGM3"],
        "OppFGA3": df["LFGA3"],
        "OppFTM": df["LFTM"],
        "OppFTA": df["LFTA"],
        "OppOR": df["LOR"],
        "OppDR": df["LDR"],
        "OppAst": df["LAst"],
        "OppTO": df["LTO"],
        "OppStl": df["LStl"],
        "OppBlk": df["LBlk"],
        "OppPF": df["LPF"],
    })
    loser_loc = df["WLoc"].map({"H": "A", "A": "H"}).fillna("N")
    loser = pd.DataFrame({
        "Season": df["Season"],
        "DayNum": df["DayNum"],
        "TeamID": df["LTeamID"],
        "OppTeamID": df["WTeamID"],
        "Score": df["LScore"],
        "OppScore": df["WScore"],
        "Win": 0,
        "Loc": loser_loc,
        "NumOT": df["NumOT"],
        "FGM": df["LFGM"],
        "FGA": df["LFGA"],
        "FGM3": df["LFGM3"],
        "FGA3": df["LFGA3"],
        "FTM": df["LFTM"],
        "FTA": df["LFTA"],
        "OR": df["LOR"],
        "DR": df["LDR"],
        "Ast": df["LAst"],
        "TO": df["LTO"],
        "Stl": df["LStl"],
        "Blk": df["LBlk"],
        "PF": df["LPF"],
        "OppFGM": df["WFGM"],
        "OppFGA": df["WFGA"],
        "OppFGM3": df["WFGM3"],
        "OppFGA3": df["WFGA3"],
        "OppFTM": df["WFTM"],
        "OppFTA": df["WFTA"],
        "OppOR": df["WOR"],
        "OppDR": df["WDR"],
        "OppAst": df["WAst"],
        "OppTO": df["WTO"],
        "OppStl": df["WStl"],
        "OppBlk": df["WBlk"],
        "OppPF": df["WPF"],
    })
    out = pd.concat([winner, loser], ignore_index=True)
    out["Poss"] = 0.5 * (
        (out["FGA"] - out["OR"] + out["TO"] + 0.475 * out["FTA"]) +
        (out["OppFGA"] - out["OppOR"] + out["OppTO"] + 0.475 * out["OppFTA"])
    )
    out["PointMargin"] = out["Score"] - out["OppScore"]
    out["FGPct"] = out["FGM"] / out["FGA"].replace(0, np.nan)
    out["FG3Pct"] = out["FGM3"] / out["FGA3"].replace(0, np.nan)
    out["FTPct"] = out["FTM"] / out["FTA"].replace(0, np.nan)
    out["eFGPct"] = (out["FGM"] + 0.5 * out["FGM3"]) / out["FGA"].replace(0, np.nan)
    out["TOVPct"] = out["TO"] / (out["FGA"] + 0.44 * out["FTA"] + out["TO"]).replace(0, np.nan)
    out["ORBRate"] = out["OR"] / (out["OR"] + out["OppDR"]).replace(0, np.nan)
    out["DRBRate"] = out["DR"] / (out["DR"] + out["OppOR"]).replace(0, np.nan)
    out["FTRate"] = out["FTA"] / out["FGA"].replace(0, np.nan)
    out["AstTORatio"] = out["Ast"] / out["TO"].replace(0, np.nan)
    return out


def _team_win_rates_from_compact(df: pd.DataFrame) -> pd.DataFrame:
    winner = pd.DataFrame({
        "Season": df["Season"],
        "TeamID": df["WTeamID"],
        "OppTeamID": df["LTeamID"],
        "Win": 1,
    })
    loser = pd.DataFrame({
        "Season": df["Season"],
        "TeamID": df["LTeamID"],
        "OppTeamID": df["WTeamID"],
        "Win": 0,
    })
    games = pd.concat([winner, loser], ignore_index=True)
    team = games.groupby(["Season", "TeamID"], as_index=False).agg(
        Games=("Win", "size"),
        Wins=("Win", "sum"),
    )
    team["WinPct"] = team["Wins"] / team["Games"].replace(0, np.nan)
    opp_strength = team[["Season", "TeamID", "WinPct"]].rename(
        columns={"TeamID": "OppTeamID", "WinPct": "OppWinPct"}
    )
    opp = games.merge(
        opp_strength,
        on=["Season", "OppTeamID"],
        how="left",
    )
    sos = opp.groupby(["Season", "TeamID"], as_index=False)["OppWinPct"].mean()
    sos = sos.rename(columns={"OppWinPct": "SOSWinPct"})
    return team.merge(sos, on=["Season", "TeamID"], how="left")


def _massey_team_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "MasseyMean", "MasseyMedian", "MasseyBest", "MasseyWorst"])
    max_day = df.groupby(["Season", "SystemName"], as_index=False)["RankingDayNum"].max()
    latest = df.merge(max_day, on=["Season", "SystemName", "RankingDayNum"], how="inner")
    out = latest.groupby(["Season", "TeamID"], as_index=False).agg(
        MasseyMean=("OrdinalRank", "mean"),
        MasseyMedian=("OrdinalRank", "median"),
        MasseyBest=("OrdinalRank", "min"),
        MasseyWorst=("OrdinalRank", "max"),
    )
    return out


def _recent_form(team_games: pd.DataFrame) -> pd.DataFrame:
    # Last-10 win rate per team/season.
    sorted_games = team_games.sort_values(["Season", "TeamID", "DayNum"])
    recent = sorted_games.groupby(["Season", "TeamID"], as_index=False).tail(10)
    return recent.groupby(["Season", "TeamID"], as_index=False).agg(
        Last10WinPct=("Win", "mean"),
        Last10PointMargin=("PointMargin", "mean"),
    )


def _build_team_season_features(
    detailed_df: pd.DataFrame,
    compact_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    massey_df: pd.DataFrame,
    gender: str,
) -> pd.DataFrame:
    team_games = _perspective_rows_from_detailed(detailed_df)
    agg = team_games.groupby(["Season", "TeamID"], as_index=False).agg(
        Games=("Win", "size"),
        Wins=("Win", "sum"),
        AvgScore=("Score", "mean"),
        AvgOppScore=("OppScore", "mean"),
        AvgPointMargin=("PointMargin", "mean"),
        AvgPoss=("Poss", "mean"),
        AvgFGPct=("FGPct", "mean"),
        AvgFG3Pct=("FG3Pct", "mean"),
        AvgFTPct=("FTPct", "mean"),
        AvgeFGPct=("eFGPct", "mean"),
        AvgTOVPct=("TOVPct", "mean"),
        AvgORBRate=("ORBRate", "mean"),
        AvgDRBRate=("DRBRate", "mean"),
        AvgFTRate=("FTRate", "mean"),
        AvgAstTORatio=("AstTORatio", "mean"),
        AvgAst=("Ast", "mean"),
        AvgTO=("TO", "mean"),
        AvgStl=("Stl", "mean"),
        AvgBlk=("Blk", "mean"),
        AvgPF=("PF", "mean"),
        AvgNumOT=("NumOT", "mean"),
    )
    agg["WinPct"] = agg["Wins"] / agg["Games"].replace(0, np.nan)

    loc_counts = (
        team_games.assign(
            IsHome=(team_games["Loc"] == "H").astype(int),
            IsAway=(team_games["Loc"] == "A").astype(int),
            IsNeutral=(team_games["Loc"] == "N").astype(int),
        )
        .groupby(["Season", "TeamID"], as_index=False)[["IsHome", "IsAway", "IsNeutral"]]
        .mean()
        .rename(columns={"IsHome": "HomeRate", "IsAway": "AwayRate", "IsNeutral": "NeutralRate"})
    )

    recent = _recent_form(team_games)
    compact_win = _team_win_rates_from_compact(compact_df)

    seed_small = seeds_df[["Season", "TeamID", "Seed"]].copy()
    seed_small["SeedNum"] = seed_small["Seed"].map(_extract_seed_number)

    conf_small = conf_df[["Season", "TeamID", "ConfAbbrev"]].drop_duplicates()
    massey = _massey_team_features(massey_df)

    out = agg.merge(loc_counts, on=["Season", "TeamID"], how="left")
    out = out.merge(recent, on=["Season", "TeamID"], how="left")
    out = out.merge(compact_win[["Season", "TeamID", "SOSWinPct"]], on=["Season", "TeamID"], how="left")
    out = out.merge(seed_small[["Season", "TeamID", "SeedNum"]], on=["Season", "TeamID"], how="left")
    out = out.merge(conf_small, on=["Season", "TeamID"], how="left")
    out = out.merge(massey, on=["Season", "TeamID"], how="left")
    out["Gender"] = gender
    return out


def _build_matchups_from_tourney_results(team_features: pd.DataFrame, tourney_df: pd.DataFrame, gender: str) -> pd.DataFrame:
    rows = tourney_df[["Season", "WTeamID", "LTeamID"]].copy()
    rows["TeamIDLow"] = rows[["WTeamID", "LTeamID"]].min(axis=1)
    rows["TeamIDHigh"] = rows[["WTeamID", "LTeamID"]].max(axis=1)
    rows["TargetLowWin"] = (rows["WTeamID"] == rows["TeamIDLow"]).astype(int)
    rows["Gender"] = gender

    feat_cols = [c for c in team_features.columns if c not in {"ConfAbbrev", "Gender"}]
    low = team_features[feat_cols].rename(columns={c: f"{c}_low" for c in feat_cols if c not in {"Season", "TeamID"}})
    high = team_features[feat_cols].rename(columns={c: f"{c}_high" for c in feat_cols if c not in {"Season", "TeamID"}})
    low = low.rename(columns={"TeamID": "TeamIDLow"})
    high = high.rename(columns={"TeamID": "TeamIDHigh"})

    out = rows.merge(low, on=["Season", "TeamIDLow"], how="left")
    out = out.merge(high, on=["Season", "TeamIDHigh"], how="left")
    out["ID"] = out["Season"].astype(str) + "_" + out["TeamIDLow"].astype(str) + "_" + out["TeamIDHigh"].astype(str)

    numeric_low_cols = [c for c in out.columns if c.endswith("_low") and np.issubdtype(out[c].dtype, np.number)]
    for low_col in numeric_low_cols:
        base = low_col[:-4]
        high_col = f"{base}_high"
        if high_col in out.columns and np.issubdtype(out[high_col].dtype, np.number):
            out[f"Diff_{base}"] = out[low_col] - out[high_col]

    out = out.dropna(subset=[c for c in out.columns if c.startswith("Diff_")] if any(c.startswith("Diff_") for c in out.columns) else None)
    return out


def _build_matchup_inference_set(team_features: pd.DataFrame, sample_submission_df: pd.DataFrame) -> pd.DataFrame:
    ids = sample_submission_df["ID"].str.split("_", expand=True)
    pair_df = pd.DataFrame({
        "Season": ids[0].astype(int),
        "TeamIDLow": ids[1].astype(int),
        "TeamIDHigh": ids[2].astype(int),
        "ID": sample_submission_df["ID"],
    })

    feat_cols = [c for c in team_features.columns if c not in {"ConfAbbrev", "Gender"}]
    low = team_features[feat_cols].rename(columns={c: f"{c}_low" for c in feat_cols if c not in {"Season", "TeamID"}})
    high = team_features[feat_cols].rename(columns={c: f"{c}_high" for c in feat_cols if c not in {"Season", "TeamID"}})
    low = low.rename(columns={"TeamID": "TeamIDLow"})
    high = high.rename(columns={"TeamID": "TeamIDHigh"})
    out = pair_df.merge(low, on=["Season", "TeamIDLow"], how="left")
    out = out.merge(high, on=["Season", "TeamIDHigh"], how="left")

    numeric_low_cols = [c for c in out.columns if c.endswith("_low") and np.issubdtype(out[c].dtype, np.number)]
    for low_col in numeric_low_cols:
        base = low_col[:-4]
        high_col = f"{base}_high"
        if high_col in out.columns and np.issubdtype(out[high_col].dtype, np.number):
            out[f"Diff_{base}"] = out[low_col] - out[high_col]
    return out


def _try_march_madness_local_bundle(goal: str) -> tuple[str | None, dict]:
    if not _is_march_madness_goal(goal):
        return None, {"step": "data_collection", "action": "march_madness_not_requested"}
    if not _MARCH_MADNESS_DIR.exists():
        return None, {"step": "data_collection", "action": "march_madness_dir_missing", "path": str(_MARCH_MADNESS_DIR)}

    try:
        csv_files = sorted(_MARCH_MADNESS_DIR.glob("*.csv"))
        refs: dict[str, str] = {}
        for p in csv_files:
            stem = p.stem
            df = _safe_read_csv(p)
            ref = f"march_mania_{stem.lower()}"
            register_dataset(ref, df, source_type="kaggle")
            refs[stem] = ref

        m_team = _build_team_season_features(
            detailed_df=get_registered_dataset(refs["MRegularSeasonDetailedResults"]),
            compact_df=get_registered_dataset(refs["MRegularSeasonCompactResults"]),
            seeds_df=get_registered_dataset(refs["MNCAATourneySeeds"]),
            conf_df=get_registered_dataset(refs["MTeamConferences"]),
            massey_df=get_registered_dataset(refs["MMasseyOrdinals"]),
            gender="M",
        )
        w_team = _build_team_season_features(
            detailed_df=get_registered_dataset(refs["WRegularSeasonDetailedResults"]),
            compact_df=get_registered_dataset(refs["WRegularSeasonCompactResults"]),
            seeds_df=get_registered_dataset(refs["WNCAATourneySeeds"]),
            conf_df=get_registered_dataset(refs["WTeamConferences"]),
            massey_df=pd.DataFrame(columns=["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]),
            gender="W",
        )
        team_frames = [f for f in [m_team, w_team] if f is not None and not f.empty]
        team_features = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame()
        team_ref = "march_mania_team_season_features"
        register_dataset(team_ref, team_features, source_type="derived")

        m_matchups = _build_matchups_from_tourney_results(
            team_features=m_team,
            tourney_df=get_registered_dataset(refs["MNCAATourneyCompactResults"]),
            gender="M",
        )
        w_matchups = _build_matchups_from_tourney_results(
            team_features=w_team,
            tourney_df=get_registered_dataset(refs["WNCAATourneyCompactResults"]),
            gender="W",
        )
        matchup_frames = [f for f in [m_matchups, w_matchups] if f is not None and not f.empty]
        train_matchups = pd.concat(matchup_frames, ignore_index=True) if matchup_frames else pd.DataFrame()
        train_ref = "march_mania_matchup_training"
        register_dataset(train_ref, train_matchups, source_type="derived")

        stage2 = get_registered_dataset(refs["SampleSubmissionStage2"])
        inference = _build_matchup_inference_set(team_features, stage2)
        inference_ref = "march_mania_matchup_inference_2026"
        register_dataset(inference_ref, inference, source_type="derived")

        return train_ref, {
            "step": "data_collection",
            "action": "march_madness_bundle_built",
            "dataset_ref": train_ref,
            "team_features_ref": team_ref,
            "inference_ref": inference_ref,
            "rows": len(train_matchups),
            "columns": list(train_matchups.columns),
            "raw_files_loaded": len(csv_files),
            "raw_refs": refs,
        }
    except Exception as exc:
        return None, {
            "step": "data_collection",
            "action": "march_madness_bundle_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _score_dataset(ref: str, goal: str = "") -> float:
    """Quality score for comparing datasets. Higher = better.

    Factors in size, completeness, feature diversity, feature–target signal,
    and goal relevance (how well column names match goal keywords).
    """
    df = get_registered_dataset(ref)
    if df is None:
        return -1.0

    n_rows = len(df)
    null_frac = df.isnull().sum().sum() / max(df.size, 1)

    useful_cols = 0
    for c in df.columns:
        nuniq = df[c].nunique()
        if nuniq <= 1:
            continue
        is_id_like = (nuniq >= n_rows * 0.95) and not np.issubdtype(df[c].dtype, np.number)
        if not is_id_like:
            useful_cols += 1

    base = n_rows * useful_cols * (1 - null_frac)

    target_col = _infer_target_column(df, goal) if goal else None
    signal = _signal_strength(df, target_col) if target_col else 0.0
    signal_bonus = 1 + signal * 2  # range [1, ~3]

    # Relevance: strongly penalise datasets whose columns don't match the goal
    relevance = _goal_relevance(df, goal) if goal else 0.5
    relevance_multiplier = 0.1 + 0.9 * relevance  # range [0.1, 1.0]

    score = base * signal_bonus * relevance_multiplier
    return score


def _try_local(goal: str, selected_model: str | None,
               linked_datasets: list[str] | None) -> tuple[str | None, dict]:
    """Try the local data-retrieval agent. Returns (ref_or_None, audit_entry)."""
    model_type = selected_model or "machine learning"
    request = (
        f"I need to train a {model_type} model for the following goal:\n"
        f"'{goal}'\n\n"
        "Please find and prepare a dataset that includes:\n"
        "- A target variable suitable for this prediction task\n"
        "- Relevant features that could help predict the target\n"
        "- Enough rows for training and validation"
    )
    if linked_datasets:
        request += f"\n\nPreferred datasets: {', '.join(linked_datasets)}"

    result = retrieve_data(request)

    if isinstance(result, DataRetrievalResult):
        ref = result.dataset_ref
        if not _validate(ref):
            return ref, {
                "step": "data_collection", "action": "local_retrieval",
                "dataset_ref": ref, "rows": result.rows,
                "columns": result.columns, "source": result.source,
                "description": result.description,
            }
        return None, {"step": "data_collection", "action": "local_validation_failed",
                       "dataset_ref": ref, "issues": _validate(ref)}

    if isinstance(result, dict) and result.get("success"):
        ref = (result.get("parsed") or {}).get("dataset_ref")
        if ref and not _validate(ref):
            return ref, {"step": "data_collection", "action": "local_retrieval",
                         "dataset_ref": ref}
        return None, {"step": "data_collection", "action": "local_partial",
                       "parsed": result.get("parsed")}

    error_msg = result.get("error", "Unknown") if isinstance(result, dict) else str(result)
    return None, {"step": "data_collection", "action": "local_failed", "error": error_msg}


def _try_curator(goal: str) -> tuple[str | None, str, dict]:
    """Try the external Dataset Curator. Returns (ref_or_None, source, audit_entry)."""
    try:
        _CURATOR_DIR = Path(__file__).parent.parent.parent / "dataset_curator"
        if str(_CURATOR_DIR.parent) not in sys.path:
            sys.path.insert(0, str(_CURATOR_DIR.parent))
        from dataset_curator import DatasetCuratorResult, curate_dataset

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()

        result = asyncio.run(curate_dataset(goal))
    except Exception as exc:
        print(f"[data_collection] Curator error: {type(exc).__name__}: {exc}")
        return None, "", {"step": "data_collection", "action": "curator_failed",
                          "error": str(exc)}

    if not isinstance(result, DatasetCuratorResult):
        return None, "", {"step": "data_collection", "action": "curator_no_result"}

    try:
        df = pd.read_csv(result.csv_path)
    except Exception as exc:
        return None, "", {"step": "data_collection", "action": "curator_csv_error",
                          "error": str(exc)}

    ref = f"curator_{Path(result.csv_path).stem}"
    register_dataset(ref, df)

    if _validate(ref):
        return None, "", {"step": "data_collection", "action": "curator_validation_failed",
                          "dataset_ref": ref, "issues": _validate(ref)}

    source = "kaggle"
    if result.hf_sources:
        source = "huggingface" if not result.kaggle_sources else "kaggle+huggingface"

    return ref, source, {
        "step": "data_collection", "action": "curator_retrieval",
        "dataset_ref": ref, "csv_path": result.csv_path,
        "rows": len(df), "columns": list(df.columns), "source": source,
        "kaggle_sources": result.kaggle_sources, "hf_sources": result.hf_sources,
        "description": result.description,
    }


# =============================================================================
# Main entry point
# =============================================================================


def data_collection(state: "TrainingAgentState") -> "TrainingAgentState":
    """Collect the best available dataset for model training.

    1. Use a pre-registered linked dataset if one exists and is valid.
    2. Search local sources via the data-retrieval agent.
    3. If ``use_external_sources`` is enabled, also search Kaggle/HuggingFace
       via the Dataset Curator and keep whichever result is better.
    """
    goal = state.get("goal", "")
    linked_datasets = state.get("linked_datasets")
    selected_model = state.get("selected_model")
    use_external = state.get("use_external_sources", False)

    audit_trace = list(state.get("audit_trace", []))
    explanations = list(state.get("explanations", []))

    if not goal:
        audit_trace.append({"step": "data_collection", "error": "No goal provided"})
        explanations.append("Data collection failed: No goal provided")
        return {**state, "collected_dataset_ref": None, "audit_trace": audit_trace,
                "explanations": explanations, "current_step": "data_collection",
                "error": "No goal provided"}

    # ----- Linked datasets (instant, no agent call) --------------------------
    if linked_datasets:
        for ref in linked_datasets:
            df = get_registered_dataset(ref)
            if df is not None and not _validate(ref):
                audit_trace.append({"step": "data_collection", "action": "used_linked_dataset",
                                    "dataset_ref": ref, "rows": len(df),
                                    "columns": list(df.columns), "source": "pre-registered"})
                explanations.append(
                    f"Using pre-registered dataset '{ref}' "
                    f"({len(df):,} rows, {len(df.columns)} columns).")
                return {**state, "collected_dataset_ref": ref,
                        "data_source": "pre-registered", "audit_trace": audit_trace,
                        "explanations": explanations, "current_step": "data_collection",
                        "error": None}

    # ----- Competition-specialized local path (March Madness) ---------------
    march_ref, march_audit = _try_march_madness_local_bundle(goal)
    audit_trace.append(march_audit)
    if march_ref and not _validate(march_ref):
        df = get_registered_dataset(march_ref)
        explanations.append(
            f"Built March Madness multi-table matchup training dataset '{march_ref}' "
            f"({len(df):,} rows, {len(df.columns)} columns) from local Kaggle bundle."
        )
        return {
            **state,
            "collected_dataset_ref": march_ref,
            "data_source": "kaggle",
            "competition_inference_ref": march_audit.get("inference_ref"),
            "competition_team_features_ref": march_audit.get("team_features_ref"),
            "audit_trace": audit_trace,
            "explanations": explanations,
            "current_step": "data_collection",
            "error": None,
        }

    # ----- Search local + (optionally) external, pick best ------------------
    local_ref, local_audit = _try_local(goal, selected_model, linked_datasets)
    audit_trace.append(local_audit)

    ext_ref, ext_source, ext_audit = None, "", {}
    if use_external:
        print("[data_collection] Also searching external sources (Kaggle + HuggingFace)...")
        ext_ref, ext_source, ext_audit = _try_curator(goal)
        audit_trace.append(ext_audit)

    # Pick the best available dataset
    best_ref, best_source = None, None
    if local_ref and ext_ref:
        local_score = _score_dataset(local_ref, goal)
        ext_score = _score_dataset(ext_ref, goal)
        if ext_score > local_score:
            best_ref, best_source = ext_ref, ext_source
            print(f"[data_collection] External dataset scored higher ({ext_score:.0f} vs {local_score:.0f}) — using '{ext_ref}'")
        else:
            best_ref, best_source = local_ref, "local"
            print(f"[data_collection] Local dataset scored higher ({local_score:.0f} vs {ext_score:.0f}) — using '{local_ref}'")
    elif local_ref:
        best_ref, best_source = local_ref, "local"
    elif ext_ref:
        best_ref, best_source = ext_ref, ext_source

    if best_ref:
        df = get_registered_dataset(best_ref)
        rows = len(df) if df is not None else "?"
        cols = len(df.columns) if df is not None else "?"
        explanations.append(
            f"Data collection complete. Using '{best_ref}' "
            f"({rows} rows, {cols} columns) from {best_source}.")
    else:
        explanations.append("Data collection failed: no suitable dataset found.")

    return {
        **state,
        "collected_dataset_ref": best_ref,
        "data_source": best_source,
        "audit_trace": audit_trace,
        "explanations": explanations,
        "current_step": "data_collection",
        "error": None if best_ref else "No suitable dataset found",
    }
