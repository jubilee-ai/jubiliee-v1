"""
Comprehensive test suite: 10 diverse tasks covering every agent capability.
Tests: selection quality, profiling, normalization, validation, joins,
row targeting, iteration on bad data, and output quality.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
import pandas as pd

CURATED_DIR = PROJECT_ROOT / "datasets" / "curated"

TESTS = [
    # --- BASIC SINGLE-DATASET ---
    {
        "id": "01", "name": "Binary clf — spam detection",
        "goal": "Build a dataset for email/SMS spam classification. I need text content and a spam/ham label.",
        "output": "t01_spam.csv", "rows": 4000,
    },
    {
        "id": "02", "name": "Regression — diamond prices",
        "goal": "Find data for predicting diamond prices. Features: carat, cut, color, clarity, depth, table, dimensions. Target: price.",
        "output": "t02_diamonds.csv", "rows": 10000,
    },
    {
        "id": "03", "name": "Multi-class — iris species",
        "goal": "Build the classic iris flower classification dataset with sepal/petal measurements and species label.",
        "output": "t03_iris.csv", "rows": None,
    },
    # --- ROW TARGETING ---
    {
        "id": "04", "name": "Exact row target — stroke prediction",
        "goal": "Build a dataset for predicting stroke occurrence. Include patient features like age, hypertension, heart disease, glucose, BMI, and a stroke label.",
        "output": "t04_stroke.csv", "rows": 3000,
    },
    {
        "id": "05", "name": "Large row target — airline satisfaction",
        "goal": "Find a dataset for predicting airline passenger satisfaction. Include features like flight distance, seat comfort, inflight service, departure delay, and a satisfaction label.",
        "output": "t05_airline.csv", "rows": 15000,
    },
    # --- MULTI-FILE / JOIN ---
    {
        "id": "06", "name": "Multi-file join — fraud train+test",
        "goal": "Find a credit card fraud detection dataset. If the download has separate train and test files, combine them into one unified set. Include transaction features and a fraud indicator.",
        "output": "t06_fraud.csv", "rows": 10000,
    },
    # --- NICHE / HARD DOMAINS ---
    {
        "id": "07", "name": "Niche — mushroom edibility",
        "goal": "Build a dataset for classifying mushrooms as edible or poisonous based on physical characteristics like cap shape, color, odor, gill size, habitat.",
        "output": "t07_mushroom.csv", "rows": None,
    },
    {
        "id": "08", "name": "Medical — heart failure",
        "goal": "Find a clinical dataset for predicting heart failure mortality. Include lab values like ejection fraction, serum creatinine, age, anemia status, and a death event label.",
        "output": "t08_heart_failure.csv", "rows": None,
    },
    # --- COMPLEX FEATURES ---
    {
        "id": "09", "name": "Regression — student performance",
        "goal": "Build a dataset for predicting student exam scores. Features should include study hours, attendance, previous scores, parental education, and a final grade as the target.",
        "output": "t09_students.csv", "rows": 5000,
    },
    {
        "id": "10", "name": "Regression — concrete strength",
        "goal": "Find a dataset for predicting concrete compressive strength. Features: cement, blast furnace slag, water, superplasticizer, coarse/fine aggregate, age. Target: compressive strength in MPa.",
        "output": "t10_concrete.csv", "rows": None,
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

    # CSV exists
    exists = Path(csv_path).exists()
    checks["csv"] = exists
    if not exists:
        notes.append("No CSV")
        checks["total"] = 0
        return checks, notes, calls

    df = pd.read_csv(csv_path, low_memory=False)

    # Row target
    desired = test.get("rows")
    if desired:
        ratio = len(df) / desired
        checks["rows"] = 0.8 <= ratio <= 1.2
        if not checks["rows"]:
            notes.append(f"Rows: {len(df):,}/{desired:,} ({ratio:.2f}x)")
    else:
        checks["rows"] = len(df) >= 30

    # Nulls
    null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0
    checks["nulls"] = null_pct < 5
    if not checks["nulls"]:
        notes.append(f"Nulls: {null_pct:.1f}%")

    # Dupes
    dupe_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
    checks["dupes"] = dupe_pct < 5
    if not checks["dupes"]:
        notes.append(f"Dupes: {dupe_pct:.1f}%")

    # Snake case
    bad = [c for c in df.columns if c != c.lower() or " " in c]
    checks["snake"] = len(bad) == 0
    if not checks["snake"]:
        notes.append(f"Non-snake: {bad[:3]}")

    # Pipeline steps
    checks["profiled"] = "profile_dataset" in names
    checks["normalized"] = "normalize_columns" in names
    checks["validated"] = "validate_target" in names
    checks["compared"] = names.count("get_dataset_info") >= 2
    checks["json"] = "csv_path" in final and "target_column" in final

    if not checks["profiled"]:
        notes.append("Skipped profile")
    if not checks["normalized"]:
        notes.append("Skipped normalize")
    if not checks["validated"]:
        notes.append("Skipped validate")
    if not checks["compared"]:
        notes.append(f"Only {names.count('get_dataset_info')} get_dataset_info (need 2+)")

    passed = sum(1 for v in checks.values() if v)
    checks["total"] = passed / len(checks)

    return checks, notes, calls


async def main():
    from agents.dataset_curator.agent import build_dataset_curator_agent

    print("=" * 80)
    print(f"  DATASET CURATOR — 10-TEST SUITE")
    print("=" * 80)

    agent, client = await build_dataset_curator_agent()
    results = []

    for t in TESTS:
        print(f"\n{'━' * 80}")
        print(f"  [{t['id']}] {t['name']}")
        print(f"{'━' * 80}")

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
        checks, notes, calls = grade(t, messages, csv_path)
        score = checks.get("total", 0)

        # Print trace
        print(f"  Time: {elapsed:.1f}s | Tools: {len(calls)} | Score: {score:.0%}")
        print(f"  Trace:")
        for i, c in enumerate(calls):
            args_s = json.dumps(c["args"], default=str)
            if len(args_s) > 100:
                args_s = args_s[:100] + "..."
            print(f"    {i+1:>2}. {c['name']:<28} {args_s}")

        # Print output summary
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"  Output: {len(df):,} rows × {len(df.columns)} cols")
            print(f"  Columns: {', '.join(df.columns[:8])}", end="")
            if len(df.columns) > 8:
                print(f" (+{len(df.columns)-8})")
            else:
                print()
        else:
            print(f"  Output: NONE")

        # Print checks
        for k, v in checks.items():
            if k == "total":
                continue
            icon = "✓" if v else "✗"
            print(f"    {icon} {k}")
        if notes:
            for n in notes:
                print(f"    ! {n}")

        results.append({
            "id": t["id"], "name": t["name"],
            "score": round(score, 2), "elapsed": round(elapsed, 1),
            "tools": len(calls), "checks": checks, "notes": notes,
        })

    # ── SUMMARY ──
    print(f"\n{'=' * 80}")
    print(f"  RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'ID':<4} {'Test':<42} {'Score':>6} {'Tools':>6} {'Time':>6}")
    print(f"  {'─'*4} {'─'*42} {'─'*6} {'─'*6} {'─'*6}")
    for r in results:
        s = r["score"]
        tools = r.get("tools", "—")
        t = r.get("elapsed", "—")
        print(f"  {r['id']:<4} {r['name']:<42} {s:>5.0%} {tools:>6} {t:>5.1f}s")

    avg = sum(r["score"] for r in results) / len(results)
    print(f"  {'─'*4} {'─'*42} {'─'*6}")
    print(f"  {'AVG':<4} {'':42} {avg:>5.0%}")

    # Per-check pass rates
    all_checks = [
        "csv", "rows", "nulls", "dupes", "snake",
        "profiled", "normalized", "validated", "compared", "json",
    ]
    print(f"\n  Per-check pass rates:")
    for ck in all_checks:
        passed = sum(1 for r in results if r.get("checks", {}).get(ck, False))
        n = len(results)
        icon = "✓" if passed == n else ("~" if passed >= n * 0.7 else "✗")
        print(f"    {icon} {ck:<20} {passed}/{n}")

    # Common issues
    all_notes = []
    for r in results:
        all_notes.extend(r.get("notes", []))
    if all_notes:
        from collections import Counter
        print(f"\n  Issues:")
        for note, cnt in Counter(all_notes).most_common():
            print(f"    [{cnt}x] {note}")

    out = PROJECT_ROOT / "agents" / "dataset_curator" / "test_suite_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results → {out}")


if __name__ == "__main__":
    asyncio.run(main())
