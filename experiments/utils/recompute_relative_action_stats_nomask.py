#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def read_vec_column(parquet_path: Path, key: str) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=[key])
    col = table[key].combine_chunks()
    arr = np.asarray(col.to_pylist(), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def q_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, eps)


def q_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [q1[..., i] for i in range(4)]
    w2, x2, y2, z2 = [q2[..., i] for i in range(4)]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w = q[..., :1]
    xyz = q[..., 1:]
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return v + 2.0 * (w * uv + uuv)


def abs_to_rel_chunks(obs0: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    # obs0: (N, 8), chunks: (N, T, 8)
    pos0 = obs0[:, None, 0:3]
    quat0 = q_normalize(obs0[:, None, 3:7])

    pos = chunks[:, :, 0:3]
    quat = q_normalize(chunks[:, :, 3:7])
    grip = chunks[:, :, 7:8]

    q0_inv = q_conj(quat0)
    rel_pos_world = pos - pos0
    rel_pos = quat_apply(q0_inv, rel_pos_world)

    rel_quat = q_mul(q0_inv, quat)
    rel_quat = q_normalize(rel_quat)

    # Canonicalize sign to match training conversion.
    flip = rel_quat[..., :1] < 0.0
    rel_quat = np.where(flip, -rel_quat, rel_quat)

    return np.concatenate([rel_pos, rel_quat, grip], axis=-1)


def compute_nomask_stats(dataset_root: Path, chunk_size: int, action_key: str) -> tuple[np.ndarray, np.ndarray, int]:
    data_root = dataset_root / 'data'
    files = sorted(data_root.glob('chunk-*/episode_*.parquet'))
    if not files:
        raise FileNotFoundError(f'No parquet files found under {data_root}')

    sum_x = None
    sum_x2 = None
    count = 0

    for i, fp in enumerate(files, 1):
        actions = read_vec_column(fp, action_key)
        states = read_vec_column(fp, 'observation.state')

        n, a = actions.shape
        if n == 0:
            continue
        if states.shape[0] != n:
            raise RuntimeError(f'row mismatch in {fp}: action={n}, state={states.shape[0]}')

        t_idx = np.arange(n, dtype=np.int64)[:, None]
        d_idx = np.arange(chunk_size, dtype=np.int64)[None, :]
        idx = t_idx + d_idx
        np.minimum(idx, n - 1, out=idx)

        chunks = actions[idx]  # (N, T, A)
        rel = abs_to_rel_chunks(states, chunks)

        rel64 = rel.astype(np.float64, copy=False)
        cur_sum = rel64.sum(axis=0)
        cur_sum2 = (rel64 * rel64).sum(axis=0)

        if sum_x is None:
            sum_x = cur_sum
            sum_x2 = cur_sum2
        else:
            sum_x += cur_sum
            sum_x2 += cur_sum2

        count += n
        if i % 50 == 0:
            print(f'[progress] {i}/{len(files)} files, frames={count}', flush=True)

    if count == 0:
        raise RuntimeError('No frames found')

    mean = sum_x / count
    var = sum_x2 / count - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32), count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--chunk-size', type=int, default=50)
    ap.add_argument('--action-key', default='action')
    ap.add_argument('--backup', action='store_true')
    args = ap.parse_args()

    ds = Path(args.dataset_root)
    out = ds / 'meta' / f'relative_action_stats_Te{args.chunk_size}.pt'
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and args.backup:
        bk = out.with_suffix(out.suffix + ".bak_" + time.strftime("%Y%m%d-%H%M%S"))
        out.rename(bk)
        print(f'[backup] {bk}')

    mean, std, frames = compute_nomask_stats(ds, args.chunk_size, args.action_key)
    obj = {'action': {'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std)}}
    torch.save(obj, out)

    print(f'[done] saved: {out}')
    print(f'[done] frames: {frames}, shape: {mean.shape}')
    print(f'[done] mean00={float(mean[0,0]):.8f}, std00={float(std[0,0]):.8f}')


if __name__ == '__main__':
    main()
