from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pint.baselines.common import baseline_spec, run_tag
from pint.query_gradients.common import timestep_count_for
from pint.training.cli import output_root


LDS_GROUP_SEEDS = (0, 1, 2)
SUBSETS_PER_SEED = 64
LDS_MODEL_COUNT = len(LDS_GROUP_SEEDS) * SUBSETS_PER_SEED
LDS_SUBSET_SIZES = {"x3": 12_500, "shape3d": 2_500, "celeba": 2_500}
LDS_TARGETS = ("simple_loss", "trajectory_deviation")
TOPK_METRICS = (
    "endpoint_mse", "trajectory_mse", "pixel_l2",
    "clip_distance", "dino_distance", "lpips",
)


def true_f_complete(path: Path, experiment: str) -> bool:
    if not path.is_file():
        return False
    expected = TOPK_METRICS if experiment == "artbench2" else LDS_TARGETS
    try:
        with np.load(path, allow_pickle=False) as payload:
            return all(name in payload.files for name in expected)
    except (OSError, ValueError):
        return False


def artbench_ranking_sign(method: str, parameter_source: str) -> int:
    """Orient stored ArtBench scores so larger always means more influential."""
    if method not in ("pint", "gas", "tracincp", "retrac"):
        raise ValueError(f"unknown ArtBench-2 method: {method}")
    if parameter_source not in ("raw", "adamw"):
        raise ValueError(f"unknown ArtBench-2 parameter source: {parameter_source}")
    return 1


def _topk_columns(scores: np.ndarray, topk: int, ranking_sign: int) -> np.ndarray:
    oriented = ranking_sign * np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(oriented)
    if int(finite.sum()) < topk:
        raise ValueError(f"fewer than {topk} finite ArtBench-2 attribution scores")
    ranking = np.where(finite, oriented, -np.inf)
    return np.argsort(-ranking, kind="stable")[:topk]


@dataclass(frozen=True)
class EvaluationJob:
    job_id: str
    experiment: str
    output_dir: Path
    train_indices: Path
    subset_id: int | None = None
    subset_seed: int | None = None
    query_id: int | None = None
    method: str | None = None
    parameter_source: str | None = None

    def as_json(self) -> dict[str, object]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self).items()
        }


def lds_universe(experiment: str) -> np.ndarray:
    if experiment == "x3":
        return np.arange(50_000, dtype=np.int64)
    path = Path(__file__).resolve().parents[3] / "dataset_manifests" / experiment / "attribution_5k_indices.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"paper LDS universe not found: {path}; run scripts/prepare_attribution_scope.py first"
        )
    return np.asarray(np.load(path), dtype=np.int64).reshape(-1)


def prepare_lds_jobs(experiment: str, root: Path) -> list[EvaluationJob]:
    universe = lds_universe(experiment)
    k = LDS_SUBSET_SIZES[experiment]
    if k > len(universe):
        raise ValueError(f"LDS subset size {k} exceeds universe size {len(universe)}")
    models = root / "lds" / "models"
    models.mkdir(parents=True, exist_ok=True)
    jobs: list[EvaluationJob] = []
    global_id = 0
    for group_seed in LDS_GROUP_SEEDS:
        group_rng = np.random.default_rng(group_seed)
        for local_subset_id in range(SUBSETS_PER_SEED):
            if experiment == "x3":
                # The pixel experiment advances one RNG directly through all
                # 64 masks in a bank; do not insert an extra derived seed.
                subset_seed = group_seed
                kept = np.sort(group_rng.choice(universe, k, replace=False)).astype(np.int64)
            else:
                # The refined JAX LDS runner records a fresh seed per subset.
                subset_seed = int(group_rng.integers(0, np.iinfo(np.int32).max))
                kept = np.sort(
                    np.random.default_rng(subset_seed).choice(universe, k, replace=False)
                ).astype(np.int64)
            model_dir = models / f"subset_{global_id:04d}"
            model_dir.mkdir(parents=True, exist_ok=True)
            indices_path = model_dir / "train_indices.npy"
            if indices_path.is_file():
                existing = np.asarray(np.load(indices_path), dtype=np.int64)
                if not np.array_equal(existing, kept):
                    raise RuntimeError(f"existing LDS subset differs: {indices_path}")
            else:
                np.save(indices_path, kept)
            metadata = {
                "subset_id": global_id,
                "subset_group_seed": group_seed,
                "local_subset_id": local_subset_id,
                "subset_seed": subset_seed,
                "subset_size": k,
                "universe_size": int(len(universe)),
            }
            (model_dir / "subset.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            jobs.append(
                EvaluationJob(
                    job_id=f"subset_{global_id:04d}",
                    experiment=experiment,
                    output_dir=model_dir,
                    train_indices=indices_path,
                    subset_id=global_id,
                    subset_seed=subset_seed,
                )
            )
            global_id += 1
    return jobs


def attribution_scores_path(experiment: str, method: str, projection_dim: int) -> Path:
    base = output_root() / experiment
    if method == "pint":
        return base / "attribution" / f"timesteps_{timestep_count_for(experiment)}" / "scores.npz"
    tag = run_tag(baseline_spec(method, experiment), projection_dim)
    return base / "methods" / method / "attribution" / tag / "scores.npz"


def prepare_artbench_jobs(
    root: Path,
    *,
    method: str,
    projection_dim: int,
    parameter_sources: tuple[str, ...],
    topk: int,
) -> list[EvaluationJob]:
    scores_path = attribution_scores_path("artbench2", method, projection_dim)
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"attribution scores not found: {scores_path}; run steps 03 and 04 for --method {method} first"
        )
    with np.load(scores_path, allow_pickle=False) as payload:
        query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
        train_positions = np.asarray(payload["train_indices"], dtype=np.int64)
        arrays = {
            "raw": np.asarray(payload["raw_scores"], dtype=np.float64),
            "adamw": (
                np.asarray(payload["adamw_scores"], dtype=np.float64)
                if "adamw_scores" in payload.files else None
            ),
        }
        if "manifest_json" in payload.files:
            score_manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
            actual_projection = int(score_manifest.get("projection_dim", projection_dim))
            if actual_projection != projection_dim:
                raise ValueError(
                    f"requested projection {projection_dim}, but scores use {actual_projection}"
                )
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("ArtBench-2 attribution contains duplicate query IDs")
    if topk > len(train_positions):
        raise ValueError(f"top-k {topk} exceeds attribution scope {len(train_positions)}")
    for source, values in arrays.items():
        if values is not None and values.shape != (len(query_ids), len(train_positions)):
            raise ValueError(f"invalid {source} score shape: {values.shape}")
    index_path = output_root().parents[0] / "data" / "artbench2" / "indices" / "idx-train.pkl"
    if not index_path.is_file():
        from pint.datasets.paths import data_root

        index_path = data_root() / "artbench2" / "indices" / "idx-train.pkl"
    with index_path.open("rb") as handle:
        selected_rows = np.asarray(list(pickle.load(handle)), dtype=np.int64)
    if train_positions.min() < 0 or train_positions.max() >= len(selected_rows):
        raise ValueError("ArtBench-2 attribution train positions are out of range")
    jobs = []
    for source in parameter_sources:
        scores = arrays[source]
        if scores is None:
            raise RuntimeError(
                f"{scores_path} has no AdamW scores; rerun step 04 without --no-adamw"
            )
        for row, query_id in enumerate(query_ids.tolist()):
            removed_columns = _topk_columns(
                scores[row], topk, artbench_ranking_sign(method, source)
            )
            removed_positions = set(train_positions[removed_columns].tolist())
            kept_rows = np.asarray(
                [value for position, value in enumerate(selected_rows) if position not in removed_positions],
                dtype=np.int64,
            )
            model_dir = root / "topk_removal" / method / source / f"q{query_id:02d}"
            model_dir.mkdir(parents=True, exist_ok=True)
            indices_path = model_dir / "train_indices.pkl"
            if indices_path.is_file():
                with indices_path.open("rb") as handle:
                    existing = np.asarray(pickle.load(handle), dtype=np.int64)
                if not np.array_equal(existing, kept_rows):
                    raise RuntimeError(f"existing removal set differs: {indices_path}")
            else:
                with indices_path.open("wb") as handle:
                    pickle.dump(kept_rows.tolist(), handle)
            jobs.append(
                EvaluationJob(
                    job_id=f"{method}_{source}_q{query_id:02d}",
                    experiment="artbench2",
                    output_dir=model_dir,
                    train_indices=indices_path,
                    query_id=query_id,
                    method=method,
                    parameter_source=source,
                )
            )
    return jobs


def merge_true_f(jobs: list[EvaluationJob], output: Path) -> Path:
    if not jobs:
        raise ValueError("cannot merge an empty evaluation run")
    records = []
    for job in jobs:
        path = job.output_dir / "true_f.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            records.append({name: np.asarray(payload[name]) for name in payload.files})
    output.parent.mkdir(parents=True, exist_ok=True)
    if jobs[0].experiment == "artbench2":
        np.savez_compressed(
            output,
            query_ids=np.asarray([job.query_id for job in jobs], dtype=np.int32),
            methods=np.asarray([job.method for job in jobs]),
            parameter_sources=np.asarray([job.parameter_source for job in jobs]),
            endpoint_mse=np.asarray([record["endpoint_mse"].item() for record in records]),
            trajectory_mse=np.asarray([record["trajectory_mse"].item() for record in records]),
            pixel_l2=np.asarray([record["pixel_l2"].item() for record in records]),
            clip_distance=np.asarray([record["clip_distance"].item() for record in records]),
            dino_distance=np.asarray([record["dino_distance"].item() for record in records]),
            lpips=np.asarray([record["lpips"].item() for record in records]),
            trajectory_reduction=np.asarray("latent_state_mse"),
        )
        return output
    query_ids = records[0]["query_ids"]
    if any(not np.array_equal(record["query_ids"], query_ids) for record in records):
        raise ValueError("evaluation models produced different query IDs")
    train_indices = [np.asarray(np.load(job.train_indices), dtype=np.int64) for job in jobs]
    np.savez_compressed(
        output,
        query_ids=query_ids,
        subset_ids=np.asarray([job.subset_id for job in jobs], dtype=np.int32),
        subset_seeds=np.asarray([job.subset_seed for job in jobs], dtype=np.int64),
        train_indices=np.stack(train_indices),
        simple_loss=np.stack([record["simple_loss"] for record in records]),
        trajectory_deviation=np.stack([
            record["trajectory_deviation"] for record in records
        ]),
        parameter_source=np.asarray("ema"),
        trajectory_reduction=np.asarray(
            "mean_snapshot_l2_squared" if jobs[0].experiment == "x3"
            else "mean_snapshot_mse"
        ),
    )
    return output
