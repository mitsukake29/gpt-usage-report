import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path(r"C:/Users/syste/GPT_usage_data")
OUTPUT_CSV = DATA_DIR / "summary_report.csv"
OUTPUT_CHART = DATA_DIR / "usage_chart.png"


SCORE_THRESHOLDS: Dict[str, List[float]] = {
    "total_input_chars": [10_000, 30_000, 70_000, 100_000],
    "total_output_chars": [10_000, 30_000, 70_000, 100_000],
    "avg_turns": [1, 3, 6, 10],
    "total_minutes": [60, 180, 420, 1_000],
    "sessions": [5, 15, 30, 60],
    "code_gen": [3, 10, 25, 50],
    "file_attachments": [1, 3, 6, 12],
    "analysis_requests": [1, 4, 10, 20],
    "tool_usage": [1, 4, 10, 20],
    "days_used": [3, 10, 20, 26],
}


def load_json_files(data_dir: Path) -> Iterable[Tuple[str, Any]]:
    for path in data_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                yield path.name, json.load(handle)
        except json.JSONDecodeError:
            continue


def ensure_iterable_threads(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "threads" in raw and isinstance(raw["threads"], list):
            return raw["threads"]
        return [raw]
    return []


def extract_user_id(thread: Dict[str, Any], file_name: str) -> str:
    for key in ("user_id", "owner_user_id", "account_user_id"):
        if thread.get(key):
            return str(thread[key])
    metadata = thread.get("metadata") or {}
    for key in ("user_id", "owner_user_id"):
        if metadata.get(key):
            return str(metadata[key])
    current_user = thread.get("current_user") or {}
    if current_user.get("id"):
        return str(current_user["id"])
    creator = thread.get("creator") or {}
    if creator.get("id"):
        return str(creator["id"])
    return Path(file_name).stem


def flatten_parts(parts: Iterable[Any]) -> List[str]:
    texts: List[str] = []
    for part in parts or []:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict):
            for key in ("text", "content", "value"):
                if isinstance(part.get(key), str):
                    texts.append(part[key])
                    break
        elif part is None:
            continue
        else:
            texts.append(str(part))
    return texts


def parse_messages(thread: Dict[str, Any]) -> List[Dict[str, Any]]:
    mapping = thread.get("mapping") or {}
    messages: List[Dict[str, Any]] = []
    for node in mapping.values():
        message = node.get("message")
        if not message:
            continue
        content = message.get("content") or {}
        messages.append(
            {
                "role": (message.get("author") or {}).get("role"),
                "parts": flatten_parts(content.get("parts") or []),
                "content_type": content.get("content_type"),
                "metadata": message.get("metadata") or {},
                "create_time": message.get("create_time"),
            }
        )
    messages.sort(key=lambda msg: msg.get("create_time") or 0)
    return messages


def compute_turns(messages: List[Dict[str, Any]]) -> int:
    turns = 0
    awaiting_assistant = False
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            awaiting_assistant = True
        elif role == "assistant" and awaiting_assistant:
            turns += 1
            awaiting_assistant = False
    return turns


def collect_usage_dates(messages: List[Dict[str, Any]]) -> Set[datetime]:
    dates: Set[datetime] = set()
    for msg in messages:
        ts = msg.get("create_time")
        if ts is None:
            continue
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except (OSError, OverflowError, TypeError):
            continue
        dates.add(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc))
    return dates


def diff_minutes(start: Optional[float], end: Optional[float]) -> float:
    if start is None or end is None:
        return 0.0
    try:
        return max(0.0, (end - start) / 60.0)
    except TypeError:
        return 0.0


def extract_text_counters(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "total_input_chars": 0,
        "total_output_chars": 0,
        "code_gen": 0,
        "file_attachments": 0,
        "analysis_requests": 0,
        "tool_usage": 0,
    }
    for msg in messages:
        role = msg.get("role")
        parts = msg.get("parts") or []
        text = "\n".join(parts)
        if role == "user":
            counts["total_input_chars"] += len(text)
        elif role == "assistant":
            counts["total_output_chars"] += len(text)
        if msg.get("content_type") == "code":
            counts["code_gen"] += 1
        attachments = msg.get("metadata", {}).get("attachments")
        if attachments:
            counts["file_attachments"] += 1
        normalized = text.lower()
        if "要約" in text or "分析" in text:
            counts["analysis_requests"] += 1
        if any(keyword.lower() in normalized for keyword in ("web", "python", "ファイル")):
            counts["tool_usage"] += 1
    return counts


def score_metric(value: float, bounds: List[float]) -> int:
    if math.isnan(value):
        return 1
    for idx, bound in enumerate(bounds, start=1):
        if value <= bound:
            return idx
    return 5


def determine_rank(average_score: float) -> str:
    if average_score >= 4.5:
        return "S"
    if average_score >= 3.5:
        return "A"
    if average_score >= 2.5:
        return "B"
    if average_score >= 1.5:
        return "C"
    return "D"


def aggregate_threads(payloads: Iterable[Tuple[str, Any]]) -> pd.DataFrame:
    aggregates: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total_input_chars": 0,
            "total_output_chars": 0,
            "total_minutes": 0.0,
            "sessions": 0,
            "code_gen": 0,
            "file_attachments": 0,
            "analysis_requests": 0,
            "tool_usage": 0,
            "turn_counts": [],
            "usage_dates": set(),
        }
    )

    for file_name, raw in payloads:
        threads = ensure_iterable_threads(raw)
        for thread in threads:
            user_id = extract_user_id(thread, file_name)
            messages = parse_messages(thread)
            if not messages:
                continue

            counters = extract_text_counters(messages)
            aggregates[user_id]["total_input_chars"] += counters["total_input_chars"]
            aggregates[user_id]["total_output_chars"] += counters["total_output_chars"]
            aggregates[user_id]["code_gen"] += counters["code_gen"]
            aggregates[user_id]["file_attachments"] += counters["file_attachments"]
            aggregates[user_id]["analysis_requests"] += counters["analysis_requests"]
            aggregates[user_id]["tool_usage"] += counters["tool_usage"]

            turn_count = compute_turns(messages)
            aggregates[user_id]["turn_counts"].append(turn_count)

            create_time = thread.get("create_time")
            update_time = thread.get("update_time")
            aggregates[user_id]["total_minutes"] += diff_minutes(create_time, update_time)
            aggregates[user_id]["sessions"] += 1

            aggregates[user_id]["usage_dates"].update(collect_usage_dates(messages))

    rows: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    rolling_window_start = now - timedelta(days=30)

    for user_id, data in aggregates.items():
        turn_counts: List[int] = data["turn_counts"]
        avg_turns = float(np.mean(turn_counts)) if turn_counts else 0.0

        usage_dates: Set[datetime] = {
            date for date in data["usage_dates"] if date >= rolling_window_start
        }
        days_used = len(usage_dates)

        metrics = {
            "total_input_chars": data["total_input_chars"],
            "total_output_chars": data["total_output_chars"],
            "avg_turns": avg_turns,
            "total_minutes": data["total_minutes"],
            "sessions": data["sessions"],
            "code_gen": data["code_gen"],
            "file_attachments": data["file_attachments"],
            "analysis_requests": data["analysis_requests"],
            "tool_usage": data["tool_usage"],
            "days_used": days_used,
        }

        scores = {
            name: score_metric(metrics[name], SCORE_THRESHOLDS[name])
            for name in SCORE_THRESHOLDS
        }
        overall_score = float(np.mean(list(scores.values()))) if scores else 0.0
        rank = determine_rank(overall_score)
        paid_flag = rank in {"S", "A", "B"}

        rows.append(
            {
                "user_id": user_id,
                **metrics,
                "score": round(overall_score, 2),
                "rank": rank,
                "paid_flag": paid_flag,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "user_id",
                "total_input_chars",
                "total_output_chars",
                "avg_turns",
                "total_minutes",
                "sessions",
                "code_gen",
                "file_attachments",
                "analysis_requests",
                "tool_usage",
                "days_used",
                "score",
                "rank",
                "paid_flag",
            ]
        )

    df = pd.DataFrame(rows)
    df.sort_values(by="score", ascending=False, inplace=True)
    return df


def generate_outputs(df: pd.DataFrame, output_csv: Path, output_chart: Path) -> None:
    if df.empty:
        print("No conversation data found. summary_report.csv was not created.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote summary report to {output_csv}")

    if output_chart.exists():
        try:
            output_chart.unlink()
        except OSError:
            pass


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    df = aggregate_threads(load_json_files(DATA_DIR))
    generate_outputs(df, OUTPUT_CSV, OUTPUT_CHART)


if __name__ == "__main__":
    main()
