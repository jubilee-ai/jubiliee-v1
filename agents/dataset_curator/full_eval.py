"""
Full evaluation: 12 diverse tests covering Kaggle-only, HF-only, dual-source,
various ML tasks, row targeting, and cross-source joining.

Grades each run on:
  - Pipeline completeness (profile, normalize, validate, compare candidates)
  - Output quality (rows, nulls, dupes, snake_case)
  - Source attribution (correct source used & reported)
  - Data reliability (columns match the task domain)
"""

import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

CURATED_DIR = PROJECT_ROOT / "datasets" / "curated"

TESTS = [
    # ── KAGGLE-PREFERRED (tabular) ──
    {
        "id": "K1", "name": "Tabular clf — wine quality",
        "goal": "Build a dataset for predicting wine quality from physicochemical properties like acidity, sugar, pH, alcohol content. Target: quality rating.",
        "output": "eval_wine.csv", "rows": 4000,
        "expect_source": "kaggle",
        "domain_cols": ["alcohol", "ph", "quality"],
    },
    {
        "id": "K2", "name": "Tabular reg — used car prices",
        "goal": "Find a dataset for predicting used car sale prices. Features: year, mileage, fuel type, transmission, engine size. Target: price.",
        "output": "eval_cars.csv", "rows": 8000,
        "expect_source": "kaggle",
        "domain_cols": ["price", "year"],
    },
    {
        "id": "K3", "name": "Tabular clf — customer churn telecom",
        "goal": "Build a dataset for predicting telecom customer churn. Include tenure, monthly charges, contract type, internet service, and a churn label.",
        "output": "eval_churn.csv", "rows": 5000,
        "expect_source": "kaggle",
        "domain_cols": ["churn"],
    },

    # ── HF-PREFERRED (NLP / benchmarks) ──
    {
        "id": "H1", "name": "NLP — tweet sentiment",
        "goal": "Build a dataset for tweet/social media sentiment analysis. I need short text and a sentiment label (positive/negative/neutral). Use a well-known benchmark.",
        "output": "eval_tweet_sent.csv", "rows": 5000,
        "expect_source": "any",
        "domain_cols": ["text"],
    },
    {
        "id": "H2", "name": "NLP — question answering",
        "goal": "Build a dataset for extractive question answering. Each row should have a context paragraph, a question, and an answer. Use a well-known QA benchmark like SQuAD.",
        "output": "eval_qa.csv", "rows": 5000,
        "expect_source": "any",
        "domain_cols": ["question"],
    },
    {
        "id": "H3", "name": "NLP — emotion classification",
        "goal": "Build a dataset for emotion classification on short texts. Labels should be emotions like joy, sadness, anger, fear, surprise, love.",
        "output": "eval_emotion.csv", "rows": 5000,
        "expect_source": "any",
        "domain_cols": ["text"],
    },

    # ── DUAL-SOURCE (agent should search both, pick best) ──
    {
        "id": "D1", "name": "Dual — spam/ham detection",
        "goal": "Build a dataset for spam vs ham classification. Search both Kaggle and HuggingFace for the best SMS or email spam dataset. Pick whichever source has higher quality data.",
        "output": "eval_spam.csv", "rows": 4000,
        "expect_source": "any",
        "domain_cols": ["text"],
    },
    {
        "id": "D2", "name": "Dual — toxic comment clf",
        "goal": "Build a dataset for detecting toxic/offensive comments. Search both Kaggle and HuggingFace. I need comment text and a toxicity label.",
        "output": "eval_toxic.csv", "rows": 8000,
        "expect_source": "any",
        "domain_cols": ["text"],
    },

    # ── CROSS-SOURCE JOIN (use datasets from both sources together) ──
    {
        "id": "X1", "name": "Cross-source — movie data enrichment",
        "goal": (
            "Build an enriched movie dataset for predicting box office revenue. "
            "Search Kaggle for a movies dataset with budget/revenue/genre info, "
            "AND search HuggingFace for a movie reviews or ratings dataset. "
            "Download one dataset from each source, then use suggest_join_keys "
            "and join_merge_tool to combine them if they share a joinable column "
            "(like movie title or ID). If they can't be joined, pick the better one."
        ),
        "output": "eval_movies_cross.csv", "rows": None,
        "expect_source": "any",
        "domain_cols": [],
    },

    # ── EDGE CASES ──
    {
        "id": "E1", "name": "Small classic — Boston housing",
        "goal": "Find the classic Boston housing / California housing dataset for regression. Target: median house value or price.",
        "output": "eval_housing.csv", "rows": None,
        "expect_source": "any",
        "domain_cols": ["price", "value", "median"],
    },
    {
        "id": "E2", "name": "Medical — diabetes prediction",
        "goal": "Build a dataset for predicting diabetes. Include features like glucose, BMI, blood pressure, insulin, age, and a diabetes outcome label.",
        "output": "eval_diabetes2.csv", "rows": None,
        "expect_source": "any",
        "domain_cols": ["glucose", "diabetes", "outcome"],
    },
    {
        "id": "E3", "name": "Time-series-like — weather forecast",
        "goal": "Find a weather/climate dataset suitable for temperature prediction. Features should include humidity, wind speed, pressure, and a temperature target.",
        "output": "eval_weather.csv", "rows": 5000,
        "expect_source": "kaggle",
        "domain_cols": ["temperature", "humidity"],
    },
]


def extract_calls(messages):
    calls = []
    for msg in messages:
        if getattr(msg, "type", "") == "ai":
            for tc in getattr(msg, "tool_calls", []):
                a = tc.get("args", {})
                compact = {}
                for k, v in a.items():
                    s = json.dumps(v, default=str)
                    compact[k] = s[:100] + "..." if len(s) > 100 else v
                calls.append({"name": tc["name"], "args": compact})
    return calls


def grade(test, messages, csv_path):
    calls = extract_calls(messages)
    names = [c["name"] for c in calls]
    final = getattr(messages[-1], "content", "") if messages else ""

    checks = {}
    notes = []
    quality = {}

    # 1. CSV exists
    exists = Path(csv_path).exists()
    checks["csv_exists"] = exists
    if not exists:
        notes.append("CSV NOT CREATED")
        checks["score"] = 0
        return checks, notes, calls, quality

    df = pd.read_csv(csv_path, low_memory=False)

    # 2. Row target
    desired = test.get("rows")
    if desired:
        ratio = len(df) / desired
        checks["row_target"] = 0.7 <= ratio <= 1.3
        if not checks["row_target"]:
            notes.append(f"Rows: {len(df):,}/{desired:,} ({ratio:.2f}x)")
    else:
        checks["row_target"] = len(df) >= 20

    # 3. Nulls
    total_cells = len(df) * len(df.columns)
    null_pct = df.isnull().sum().sum() / total_cells * 100 if total_cells > 0 else 0
    checks["low_nulls"] = null_pct < 5
    if not checks["low_nulls"]:
        notes.append(f"Nulls: {null_pct:.1f}%")
    quality["null_pct"] = round(null_pct, 2)

    # 4. Dupes
    dupe_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
    checks["low_dupes"] = dupe_pct < 5
    if not checks["low_dupes"]:
        notes.append(f"Dupes: {dupe_pct:.1f}%")
    quality["dupe_pct"] = round(dupe_pct, 2)

    # 5. Snake case
    bad_cols = [c for c in df.columns if c != c.lower() or " " in c]
    checks["snake_case"] = len(bad_cols) == 0
    if not checks["snake_case"]:
        notes.append(f"Non-snake: {bad_cols[:3]}")

    # 6. Pipeline steps
    checks["profiled"] = "profile_dataset" in names
    checks["normalized"] = "normalize_columns" in names
    checks["validated"] = "validate_target" in names
    searched_kaggle = "search_datasets" in names
    searched_hf = "hub_repo_search" in names
    checks["searched_both"] = searched_kaggle and searched_hf
    checks["compared_candidates"] = (
        names.count("get_dataset_info") + names.count("hub_repo_details") >= 2
    )

    # 7. Source usage
    used_kaggle_dl = "download_kaggle_dataset" in names
    used_hf_dl = "download_hf_dataset" in names
    checks["used_kaggle"] = used_kaggle_dl
    checks["used_hf"] = used_hf_dl
    checks["used_both_sources"] = used_kaggle_dl and used_hf_dl

    # 8. JSON output
    checks["json_output"] = "csv_path" in final and "target_column" in final

    # 9. Domain relevance — at least one expected column present (fuzzy)
    domain_cols = test.get("domain_cols", [])
    if domain_cols:
        col_str = " ".join(df.columns).lower()
        found = [c for c in domain_cols if c in col_str]
        checks["domain_match"] = len(found) > 0
        if not checks["domain_match"]:
            notes.append(f"No domain cols found. Expected: {domain_cols}, got: {list(df.columns)[:8]}")
    else:
        checks["domain_match"] = True

    # 10. Data reliability — basic stats
    quality["rows"] = len(df)
    quality["cols"] = len(df.columns)
    quality["dtypes"] = dict(df.dtypes.astype(str).value_counts())

    # Numeric column ranges (sanity check — no infinities, no all-zero)
    num_cols = df.select_dtypes(include=[np.number]).columns
    bad_numerics = []
    for c in num_cols:
        if df[c].replace([np.inf, -np.inf], np.nan).isnull().all():
            bad_numerics.append(f"{c}: all inf/nan")
        elif df[c].std() == 0 and len(df) > 10:
            bad_numerics.append(f"{c}: zero variance")
    if bad_numerics:
        notes.append(f"Bad numerics: {bad_numerics[:3]}")
    quality["bad_numerics"] = bad_numerics

    if not checks["profiled"]:
        notes.append("Skipped profile_dataset")
    if not checks["normalized"]:
        notes.append("Skipped normalize_columns")
    if not checks["validated"]:
        notes.append("Skipped validate_target")
    if not checks["searched_both"]:
        if not searched_kaggle:
            notes.append("Did not search Kaggle")
        if not searched_hf:
            notes.append("Did not search HuggingFace")

    passed = sum(1 for k, v in checks.items() if k != "score" and v)
    total = sum(1 for k in checks if k != "score")
    checks["score"] = round(passed / total, 2) if total > 0 else 0

    return checks, notes, calls, quality


async def main():
    from agents.dataset_curator.agent import build_dataset_curator_agent

    print("=" * 90)
    print("  FULL EVALUATION — 12 tests (Kaggle / HuggingFace / Dual / Cross-source)")
    print("=" * 90)

    agent, client = await build_dataset_curator_agent()
    results = []

    for t in TESTS:
        print(f"\n{'━' * 90}")
        print(f"  [{t['id']}] {t['name']}")
        print(f"{'━' * 90}")

        msg = t["goal"]
        msg += f"\n\nPlease name the output CSV: {t['output']}"
        if t["rows"]:
            msg += (
                f"\n\nThe final dataset should have approximately {t['rows']:,} rows. "
                f"Use sample_rows (random) to downsample if the source is larger, "
                f"or note if the source has fewer rows than requested."
            )

        start = time.time()
        try:
            res = await agent.ainvoke({"messages": [{"role": "user", "content": msg}]})
            elapsed = time.time() - start
            messages = res.get("messages", [])
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR ({elapsed:.0f}s): {e}")
            results.append({"id": t["id"], "name": t["name"], "score": 0, "error": str(e)})
            continue

        csv_path = str(CURATED_DIR / t["output"])
        checks, notes, calls, quality = grade(t, messages, csv_path)
        score = checks.get("score", 0)

        # ── Trace ──
        print(f"  Time: {elapsed:.1f}s | Tools: {len(calls)} | Score: {score:.0%}")
        print(f"\n  Trace:")
        for i, c in enumerate(calls):
            args_s = json.dumps(c["args"], default=str)
            if len(args_s) > 100:
                args_s = args_s[:100] + "..."
            print(f"    {i+1:>2}. {c['name']:<28} {args_s}")

        # ── Source usage ──
        sources = []
        if checks.get("used_kaggle"):
            sources.append("Kaggle")
        if checks.get("used_hf"):
            sources.append("HuggingFace")
        print(f"\n  Sources downloaded from: {', '.join(sources) or 'NONE'}")
        print(f"  Searched both: {'Yes' if checks.get('searched_both') else 'No'}")

        # ── Output summary ──
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"\n  Output: {len(df):,} rows × {len(df.columns)} cols")
            print(f"  Columns: {', '.join(df.columns[:10])}", end="")
            if len(df.columns) > 10:
                print(f" (+{len(df.columns) - 10})")
            else:
                print()
            print(f"  Null%: {quality.get('null_pct', '?')}% | Dupe%: {quality.get('dupe_pct', '?')}%")
            if quality.get("bad_numerics"):
                print(f"  Bad numerics: {quality['bad_numerics']}")

            # Show sample values for reliability check
            print(f"\n  Sample data (first 3 rows):")
            for _, row in df.head(3).iterrows():
                vals = [f"{c}={repr(v)[:40]}" for c, v in row.items()]
                print(f"    {' | '.join(vals[:4])}")
        else:
            print(f"\n  Output: NONE")

        # ── Check results ──
        print(f"\n  Checks:")
        for k, v in checks.items():
            if k == "score":
                continue
            icon = "PASS" if v else "FAIL"
            print(f"    [{icon}] {k}")
        if notes:
            print(f"\n  Issues:")
            for n in notes:
                print(f"    ! {n}")

        results.append({
            "id": t["id"], "name": t["name"],
            "score": score, "elapsed": round(elapsed, 1),
            "tools": len(calls), "checks": checks, "notes": notes,
            "quality": quality,
            "sources_used": sources,
        })

    # ═══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'ID':<5} {'Test':<40} {'Score':>6} {'Source':>14} {'Time':>6}")
    print(f"  {'─'*5} {'─'*40} {'─'*6} {'─'*14} {'─'*6}")
    for r in results:
        s = r["score"]
        src = "+".join(r.get("sources_used", ["?"])) or "?"
        t = r.get("elapsed", 0)
        print(f"  {r['id']:<5} {r['name']:<40} {s:>5.0%} {src:>14} {t:>5.1f}s")

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"  {'─'*5} {'─'*40} {'─'*6}")
    print(f"  {'AVG':<5} {'':40} {avg:>5.0%}")

    # Per-check pass rates
    all_check_names = [
        "csv_exists", "row_target", "low_nulls", "low_dupes", "snake_case",
        "profiled", "normalized", "validated", "searched_both",
        "compared_candidates", "json_output", "domain_match",
    ]
    print(f"\n  Per-check pass rates:")
    for ck in all_check_names:
        passed = sum(1 for r in results if r.get("checks", {}).get(ck, False))
        n = len(results)
        icon = "PASS" if passed == n else ("WARN" if passed >= n * 0.7 else "FAIL")
        print(f"    [{icon}] {ck:<25} {passed}/{n}")

    # Source distribution
    src_counts = Counter()
    for r in results:
        for s in r.get("sources_used", []):
            src_counts[s] += 1
    print(f"\n  Source distribution:")
    for src, cnt in src_counts.most_common():
        print(f"    {src}: {cnt}/{len(results)} tests")

    both_count = sum(1 for r in results if len(r.get("sources_used", [])) >= 2)
    print(f"    Both in same run: {both_count}/{len(results)} tests")

    # Common issues
    all_notes = []
    for r in results:
        all_notes.extend(r.get("notes", []))
    if all_notes:
        print(f"\n  Issues:")
        for note, cnt in Counter(all_notes).most_common():
            print(f"    [{cnt}x] {note}")

    out = PROJECT_ROOT / "agents" / "dataset_curator" / "full_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results → {out}")


if __name__ == "__main__":
    asyncio.run(main())
