from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pint.baselines.common import BASELINE_METHODS
from pint.evaluation.common import (
    LDS_TARGETS, TOPK_METRICS, attribution_scores_path,
)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 2:
        return float("nan")
    left_rank = _average_ranks(left[finite])
    right_rank = _average_ranks(right[finite])
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return float(np.dot(left_rank, right_rank) / denominator) if denominator else float("nan")


def _membership(score_indices: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    positions = {int(value): index for index, value in enumerate(score_indices.tolist())}
    matrix = np.zeros((len(subsets), len(score_indices)), dtype=np.float64)
    for row, subset in enumerate(subsets):
        try:
            columns = [positions[int(value)] for value in subset]
        except KeyError as exc:
            raise ValueError(
                f"LDS subset index {exc.args[0]} is absent from the attribution scope"
            ) from exc
        matrix[row, columns] = 1.0
    return matrix


def lds_query_rows(
    experiment: str, true_f_path: Path, *, projection_dim: int
) -> list[dict[str, object]]:
    from tqdm.auto import tqdm

    with np.load(true_f_path, allow_pickle=False) as payload:
        query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
        subsets = np.asarray(payload["train_indices"], dtype=np.int64)
        targets = {name: np.asarray(payload[name], dtype=np.float64) for name in LDS_TARGETS}
    rows: list[dict[str, object]] = []
    prediction_sign = -1.0
    score_inputs = []
    for method in ("pint", *BASELINE_METHODS):
        score_path = attribution_scores_path(experiment, method, projection_dim)
        if not score_path.is_file():
            continue
        with np.load(score_path, allow_pickle=False) as payload:
            score_query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
            score_indices = np.asarray(payload["train_indices"], dtype=np.int64)
            score_sets = {"raw": np.asarray(payload["raw_scores"], dtype=np.float64)}
            if "adamw_scores" in payload.files:
                score_sets["adamw"] = np.asarray(payload["adamw_scores"], dtype=np.float64)
        if set(score_query_ids.tolist()) != set(query_ids.tolist()):
            raise ValueError(f"query IDs differ between {score_path} and {true_f_path}")
        score_inputs.append((method, score_query_ids, score_indices, score_sets))
    if not score_inputs:
        raise FileNotFoundError(
            f"no attribution scores are available for {experiment}; run step 04 first"
        )

    total = sum(
        len(score_sets) * len(targets) * len(query_ids)
        for _, _, _, score_sets in score_inputs
    )
    progress = tqdm(
        total=total,
        desc=f"{experiment} LDS",
        unit="query-metric",
        dynamic_ncols=True,
    )
    try:
        for method, score_query_ids, score_indices, score_sets in score_inputs:
            query_positions = {
                int(value): index for index, value in enumerate(score_query_ids)
            }
            membership = _membership(score_indices, subsets)
            for source, scores in score_sets.items():
                predictions = prediction_sign * (membership @ scores.T)
                for target_name, target_values in targets.items():
                    for target_column, query_id in enumerate(query_ids.tolist()):
                        score_column = query_positions[int(query_id)]
                        rows.append({
                            "experiment": experiment,
                            "evaluation": "lds_spearman",
                            "method": method,
                            "parameter_source": source,
                            "target": target_name,
                            "query_id": int(query_id),
                            "value": finite_spearman(
                                predictions[:, score_column],
                                target_values[:, target_column],
                            ),
                            "prediction_sign": prediction_sign,
                        })
                        progress.update(1)
                        progress.set_postfix(
                            method=method, source=source, target=target_name,
                            refresh=False,
                        )
    finally:
        progress.close()
    return rows


def topk_query_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "topk_removal").glob("*/true_f.npz")):
        with np.load(path, allow_pickle=False) as payload:
            query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
            methods = np.asarray(payload["methods"]).astype(str)
            sources = np.asarray(payload["parameter_sources"]).astype(str)
            metrics = {name: np.asarray(payload[name], dtype=np.float64) for name in TOPK_METRICS}
        for index, query_id in enumerate(query_ids.tolist()):
            for metric, values in metrics.items():
                rows.append({
                    "experiment": "artbench2",
                    "evaluation": "topk_removal",
                    "method": str(methods[index]),
                    "parameter_source": str(sources[index]),
                    "target": metric,
                    "query_id": int(query_id),
                    "value": float(values[index]),
                    "prediction_sign": "",
                })
    if not rows:
        raise FileNotFoundError(f"no merged top-k results found under {root / 'topk_removal'}")
    return rows


def write_reports(
    experiment: str, root: Path, *, projection_dim: int = 4096
) -> tuple[Path, Path]:
    rows = (
        topk_query_rows(root)
        if experiment == "artbench2"
        else lds_query_rows(
            experiment, root / "lds" / "true_f.npz", projection_dim=projection_dim
        )
    )
    report_root = root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "query_metrics.csv"
    fieldnames = (
        "experiment", "evaluation", "method", "parameter_source", "target",
        "query_id", "value", "prediction_sign",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["parameter_source"]), str(row["target"]))
        groups.setdefault(key, []).append(float(row["value"]))
    lines = [
        f"{experiment} evaluation report",
        f"query_csv: {csv_path}",
        "",
        "method | parameter_source | target | n | mean | std",
    ]
    for (method, source, target), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        mean = float(np.mean(finite)) if len(finite) else float("nan")
        std = float(np.std(finite)) if len(finite) else float("nan")
        lines.append(
            f"{method} | {source} | {target} | {len(finite)} | "
            f"{mean:.8f} | {std:.8f}"
        )
    report_path = report_root / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, csv_path
