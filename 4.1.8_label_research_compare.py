import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "4.1.8"


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_symbol(symbol: str):
    return str(symbol).strip().upper()


def extract_timestamp_from_name(path: Path):
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        maybe = parts[-2] + "_" + parts[-1]
        try:
            return datetime.strptime(maybe, "%Y%m%d_%H%M%S")
        except Exception:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return datetime.min


def find_candidate_jsons(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.json"))


def json_matches_symbol(data, symbol: str):
    target = normalize_symbol(symbol)

    selected_symbol = safe_get(data, "selected_symbol")
    if selected_symbol and normalize_symbol(selected_symbol) == target:
        return True

    universe = safe_get(data, "universe", default=[])
    if isinstance(universe, list) and target in [normalize_symbol(x) for x in universe]:
        return True

    symbol_field = safe_get(data, "symbol")
    if symbol_field and normalize_symbol(symbol_field) == target:
        return True

    return False


def choose_latest_matching_json(folder: Path, symbol: str, expected_version: str = None):
    candidates = []
    for p in find_candidate_jsons(folder):
        try:
            data = load_json(p)
        except Exception:
            continue

        if not json_matches_symbol(data, symbol):
            continue

        version = str(data.get("version", "")).strip()
        if expected_version and version != expected_version:
            continue

        ts = extract_timestamp_from_name(p)
        candidates.append((ts, p, data))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_path, best_data = candidates[0]
    return best_path, best_data


def choose_latest_matching_json_multi(folders, symbol: str, expected_version: str = None):
    candidates = []

    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue

        for p in find_candidate_jsons(folder):
            try:
                data = load_json(p)
            except Exception:
                continue

            if not json_matches_symbol(data, symbol):
                continue

            version = str(data.get("version", "")).strip()
            if expected_version and version != expected_version:
                continue

            ts = extract_timestamp_from_name(p)
            candidates.append((ts, p, data))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_path, best_data = candidates[0]
    return best_path, best_data


def summarize_run(data, file_path: Path):
    stats = data.get("best_backtest_stats", {}) or {}
    params = data.get("best_strategy_params", {}) or {}
    model_params = data.get("model_params", {}) or {}
    feature_cols = data.get("feature_cols", []) or []
    walkforward = data.get("walkforward", {}) or {}
    data_window = data.get("data_window", {}) or {}
    market_hours = data.get("market_hours", {}) or {}
    selection_reason = data.get("selection_reason", {}) or {}

    summary = {
        "version": data.get("version"),
        "file_path": str(file_path),
        "selected_symbol": data.get("selected_symbol"),
        "selection_mode": data.get("selection_mode"),
        "primary_metric": selection_reason.get("primary_metric"),
        "selection_summary": selection_reason.get("summary"),
        "pr_auc": stats.get("pr_auc"),
        "avg_daily_pnl": stats.get("avg_daily_pnl"),
        "median_daily_pnl": stats.get("median_daily_pnl"),
        "worst_day_pnl": stats.get("worst_day_pnl"),
        "best_day_pnl": stats.get("best_day_pnl"),
        "avg_daily_return_on_account": stats.get("avg_daily_return_on_account"),
        "avg_daily_return_on_notional": stats.get("avg_daily_return_on_notional"),
        "median_daily_return_on_notional": stats.get("median_daily_return_on_notional"),
        "worst_day_return_on_notional": stats.get("worst_day_return_on_notional"),
        "best_day_return_on_notional": stats.get("best_day_return_on_notional"),
        "positive_day_ratio": stats.get("positive_day_ratio"),
        "total_pnl": stats.get("total_pnl"),
        "total_trades": stats.get("total_trades"),
        "avg_trades_per_day": stats.get("avg_trades_per_day"),
        "avg_daily_notional_used": stats.get("avg_daily_notional_used"),
        "median_daily_notional_used": stats.get("median_daily_notional_used"),
        "avg_entry_notional": stats.get("avg_entry_notional"),
        "eval_days_used": stats.get("eval_days_used"),
        "train_days": stats.get("train_days"),
        "equity_final": stats.get("equity_final"),
        "horizon": params.get("horizon"),
        "tp": params.get("tp"),
        "sl": params.get("sl"),
        "p_enter": params.get("p_enter"),
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "model_params": model_params,
        "walkforward_train_days": walkforward.get("train_days"),
        "walkforward_test_days": walkforward.get("test_days"),
        "walkforward_eval_days": walkforward.get("eval_days"),
        "walkforward_step_days": walkforward.get("step_days"),
        "market_open": market_hours.get("open"),
        "market_close": market_hours.get("close"),
        "data_start": data_window.get("start"),
        "data_end": data_window.get("end"),
        "calendar_days": data_window.get("calendar_days"),
        "feed": data_window.get("feed"),
        "qty": data.get("qty"),
        "initial_cash": data.get("initial_cash"),
    }
    return summary


def compute_delta(before_val, after_val):
    if before_val is None or after_val is None:
        return None
    try:
        return after_val - before_val
    except Exception:
        return None


def pct_change(before_val, after_val):
    if before_val in [None, 0] or after_val is None:
        return None
    try:
        return (after_val - before_val) / abs(before_val)
    except Exception:
        return None


def compare_runs(before_summary, after_summary):
    compare_metrics = [
        "pr_auc",
        "avg_daily_pnl",
        "median_daily_pnl",
        "worst_day_pnl",
        "best_day_pnl",
        "avg_daily_return_on_account",
        "avg_daily_return_on_notional",
        "median_daily_return_on_notional",
        "worst_day_return_on_notional",
        "best_day_return_on_notional",
        "positive_day_ratio",
        "total_pnl",
        "total_trades",
        "avg_trades_per_day",
        "avg_daily_notional_used",
        "median_daily_notional_used",
        "avg_entry_notional",
        "eval_days_used",
        "train_days",
        "equity_final",
        "horizon",
        "tp",
        "sl",
        "p_enter",
        "feature_count",
    ]

    rows = []
    for metric in compare_metrics:
        before_val = before_summary.get(metric)
        after_val = after_summary.get(metric)
        rows.append(
            {
                "metric": metric,
                "before_416_highconviction": before_val,
                "after_417_featureboost": after_val,
                "delta_after_minus_before": compute_delta(before_val, after_val),
                "pct_change_after_vs_before": pct_change(before_val, after_val),
            }
        )

    df = pd.DataFrame(rows)

    before_features = set(before_summary.get("feature_cols", []) or [])
    after_features = set(after_summary.get("feature_cols", []) or [])

    feature_compare = pd.DataFrame(
        [
            {"feature": f, "status": "kept"}
            for f in sorted(before_features & after_features)
        ]
        + [
            {"feature": f, "status": "removed_in_417"}
            for f in sorted(before_features - after_features)
        ]
        + [
            {"feature": f, "status": "added_in_417"}
            for f in sorted(after_features - before_features)
        ]
    )

    interpretation = {
        "better_on_avg_daily_return_on_notional": (
            after_summary.get("avg_daily_return_on_notional", float("-inf"))
            > before_summary.get("avg_daily_return_on_notional", float("-inf"))
        ),
        "better_on_total_pnl": (
            after_summary.get("total_pnl", float("-inf"))
            > before_summary.get("total_pnl", float("-inf"))
        ),
        "better_on_positive_day_ratio": (
            after_summary.get("positive_day_ratio", float("-inf"))
            > before_summary.get("positive_day_ratio", float("-inf"))
        ),
        "worse_on_worst_day_pnl": (
            after_summary.get("worst_day_pnl", float("inf"))
            < before_summary.get("worst_day_pnl", float("inf"))
        ),
        "more_trades": (
            after_summary.get("total_trades", float("-inf"))
            > before_summary.get("total_trades", float("-inf"))
        ),
        "fewer_trades": (
            after_summary.get("total_trades", float("inf"))
            < before_summary.get("total_trades", float("inf"))
        ),
        "more_features_in_417": len(after_features) > len(before_features),
        "added_feature_count": len(after_features - before_features),
        "removed_feature_count": len(before_features - after_features),
    }

    return df, feature_compare, interpretation


def print_run_summary(title, summary):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    keys = [
        "version",
        "file_path",
        "selected_symbol",
        "selection_mode",
        "pr_auc",
        "avg_daily_return_on_notional",
        "total_pnl",
        "positive_day_ratio",
        "total_trades",
        "worst_day_pnl",
        "best_day_pnl",
        "horizon",
        "tp",
        "sl",
        "p_enter",
        "feature_count",
    ]
    for k in keys:
        print(f"{k}: {summary.get(k)}")


def print_compare_highlights(compare_df):
    print("\n" + "=" * 100)
    print("KEY METRIC COMPARISON")
    print("=" * 100)
    focus = compare_df[
        compare_df["metric"].isin(
            [
                "pr_auc",
                "avg_daily_return_on_notional",
                "total_pnl",
                "positive_day_ratio",
                "total_trades",
                "worst_day_pnl",
                "best_day_pnl",
                "feature_count",
            ]
        )
    ]
    print(focus.to_string(index=False))


def save_outputs(
    out_dir: Path,
    symbol: str,
    before_path: Path,
    after_path: Path,
    before_summary: dict,
    after_summary: dict,
    compare_df: pd.DataFrame,
    feature_compare_df: pd.DataFrame,
    interpretation: dict,
):
    ts = now_stamp()
    ensure_dir(out_dir)

    summary_json = out_dir / f"label_compare_summary_{symbol}_{ts}.json"
    compare_csv = out_dir / f"label_compare_metrics_{symbol}_{ts}.csv"
    feature_csv = out_dir / f"label_compare_features_{symbol}_{ts}.csv"

    payload = {
        "version": VERSION,
        "symbol": symbol,
        "before_label": "4.1.6_highconviction",
        "after_label": "4.1.7_featureboost",
        "before_file": str(before_path),
        "after_file": str(after_path),
        "before_summary": before_summary,
        "after_summary": after_summary,
        "interpretation": interpretation,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    compare_df.to_csv(compare_csv, index=False)
    feature_compare_df.to_csv(feature_csv, index=False)

    print("\nSaved outputs:")
    print(summary_json)
    print(compare_csv)
    print(feature_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)

    parser.add_argument("--before_json", type=str, default=None)
    parser.add_argument("--after_json", type=str, default=None)

    parser.add_argument("--before_dir", type=str, default="selector_out_416")
    parser.add_argument("--after_dir", type=str, default="selector_out_417")
    parser.add_argument("--out_dir", type=str, default="selector_out_418")

    parser.add_argument("--before_version", type=str, default="4.1.6")
    parser.add_argument("--after_version", type=str, default="4.1.7")

    args = parser.parse_args()

    symbol = normalize_symbol(args.symbol)
    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)
    out_dir = Path(args.out_dir)

    print(f"Symbol: {symbol}")
    print(f"Searching BEFORE dir: {before_dir}")
    print(f"Searching AFTER dir: {after_dir}")

    if args.before_json:
        before_path = Path(args.before_json)
        before_data = load_json(before_path)
    else:
        before_path, before_data = choose_latest_matching_json_multi(
            folders=[before_dir, "selector_out_416", "selector_out_415", "selector_out_417"],
            symbol=symbol,
            expected_version=args.before_version,
        )

    if args.after_json:
        after_path = Path(args.after_json)
        after_data = load_json(after_path)
    else:
        after_path, after_data = choose_latest_matching_json_multi(
            folders=[after_dir, "selector_out_417", "selector_out_415", "selector_out_416"],
            symbol=symbol,
            expected_version=args.after_version,
        )

    if before_data is None:
        raise FileNotFoundError(
            f"Could not find a matching {args.before_version} JSON for symbol={symbol}. "
            f"Try --before_json with the exact best_etf_selection_*.json path."
        )

    if after_data is None:
        raise FileNotFoundError(
            f"Could not find a matching {args.after_version} JSON for symbol={symbol}. "
            f"Try --after_json with the exact best_etf_selection_*.json path."
        )

    before_summary = summarize_run(before_data, before_path)
    after_summary = summarize_run(after_data, after_path)

    print_run_summary("BEFORE | 4.1.6 HIGHCONVICTION", before_summary)
    print_run_summary("AFTER | 4.1.7 FEATUREBOOST", after_summary)

    compare_df, feature_compare_df, interpretation = compare_runs(
        before_summary, after_summary
    )

    print_compare_highlights(compare_df)

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    for k, v in interpretation.items():
        print(f"{k}: {v}")

    save_outputs(
        out_dir=out_dir,
        symbol=symbol,
        before_path=before_path,
        after_path=after_path,
        before_summary=before_summary,
        after_summary=after_summary,
        compare_df=compare_df,
        feature_compare_df=feature_compare_df,
        interpretation=interpretation,
    )


if __name__ == "__main__":
    main()