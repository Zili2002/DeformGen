from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _to_numpy_points(points_world: Any) -> np.ndarray:
    if hasattr(points_world, "detach"):
        points_world = points_world.detach().cpu().numpy()
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_world must be (N,>=2), got {pts.shape}")
    return pts[:, :2]


def _xy_mask_from_points(
    points_xy: np.ndarray,
    grid_size: int,
    pad_ratio: float,
    point_radius_px: int,
    close_kernel: int,
    open_kernel: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mins = points_xy.min(axis=0)
    maxs = points_xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    origin = mins - pad_ratio * span
    full_span = span * (1.0 + 2.0 * pad_ratio)
    denom = float(max(full_span[0], full_span[1], 1e-6))
    scale = float((grid_size - 1) / denom)

    pix = np.round((points_xy - origin) * scale).astype(np.int32)
    pix = np.clip(pix, 0, grid_size - 1)

    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if point_radius_px <= 0:
        mask[pix[:, 1], pix[:, 0]] = 255
    else:
        for x, y in pix:
            cv2.circle(mask, (int(x), int(y)), int(point_radius_px), 255, -1)

    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    meta = {
        "grid_size": int(grid_size),
        "origin_xy": [float(origin[0]), float(origin[1])],
        "scale": float(scale),
        "point_radius_px": int(point_radius_px),
        "close_kernel": int(close_kernel),
        "open_kernel": int(open_kernel),
    }
    return mask, meta


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = np.zeros_like(mask)
    comp[labels == largest_idx] = 255
    return comp


def evaluate_cloth3_last_frame_triangle(
    points_world: Any,
    cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    points_xy = _to_numpy_points(points_world)
    result: Dict[str, Any] = {
        "mode": "last_frame_triangle",
        "success": False,
        "num_points": int(points_xy.shape[0]),
    }
    if points_xy.shape[0] < 3:
        result["reason"] = "too_few_points"
        return result

    get_cfg = (lambda k, d: d) if cfg is None else (lambda k, d: cfg.get(k, d))
    grid_size = int(get_cfg("cloth3_success_grid_size", 256))
    pad_ratio = float(get_cfg("cloth3_success_pad_ratio", 0.08))
    point_radius_px = int(get_cfg("cloth3_success_point_radius_px", 2))
    close_kernel = int(get_cfg("cloth3_success_close_kernel", 5))
    open_kernel = int(get_cfg("cloth3_success_open_kernel", 3))
    approx_eps_ratio = float(get_cfg("cloth3_success_approx_eps_ratio", 0.03))

    iou_thresh = float(get_cfg("cloth3_success_iou_thresh", 0.72))
    cover_thresh = float(get_cfg("cloth3_success_cover_thresh", 0.80))
    vmin = int(get_cfg("cloth3_success_vertices_min", 3))
    vmax = int(get_cfg("cloth3_success_vertices_max", 4))

    mask_all, mask_meta = _xy_mask_from_points(
        points_xy=points_xy,
        grid_size=grid_size,
        pad_ratio=pad_ratio,
        point_radius_px=point_radius_px,
        close_kernel=close_kernel,
        open_kernel=open_kernel,
    )
    mask = _largest_component(mask_all)
    area_mask = int(np.count_nonzero(mask))
    if area_mask < 10:
        result.update(mask_meta)
        result["reason"] = "mask_too_small"
        return result

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        result.update(mask_meta)
        result["reason"] = "no_contour"
        return result
    contour = max(contours, key=cv2.contourArea)

    perimeter = float(cv2.arcLength(contour, True))
    eps = max(1e-6, approx_eps_ratio * perimeter)
    approx = cv2.approxPolyDP(contour, eps, True)
    vertex_count = int(len(approx))
    vertex_ok = bool(vmin <= vertex_count <= vmax)

    tri_area_float, tri_pts = cv2.minEnclosingTriangle(contour)
    if tri_pts is None or tri_area_float <= 1e-8:
        result.update(mask_meta)
        result["reason"] = "triangle_fit_failed"
        return result

    tri_pts = np.round(tri_pts.reshape(3, 2)).astype(np.int32)
    tri_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(tri_mask, tri_pts, 255)

    inter = int(np.count_nonzero((mask > 0) & (tri_mask > 0)))
    union = int(np.count_nonzero((mask > 0) | (tri_mask > 0)))
    tri_area_px = int(np.count_nonzero(tri_mask))
    iou_tri = float(inter / union) if union > 0 else 0.0
    cover = float(area_mask / max(tri_area_px, 1))
    score = float(0.7 * iou_tri + 0.3 * cover)
    success = bool(vertex_ok and iou_tri >= iou_thresh and cover >= cover_thresh)

    result.update(mask_meta)
    result.update(
        {
            "success": success,
            "iou_tri": iou_tri,
            "cover": cover,
            "score": score,
            "vertex_count": vertex_count,
            "vertex_ok": vertex_ok,
            "area_mask_px": area_mask,
            "area_triangle_px": tri_area_px,
            "iou_thresh": iou_thresh,
            "cover_thresh": cover_thresh,
            "vertices_min": vmin,
            "vertices_max": vmax,
            "triangle_xy_pixel": tri_pts.astype(float).tolist(),
        }
    )
    return result


def save_cloth3_triangle_debug_image(
    points_world: Any,
    result: Dict[str, Any],
    output_path: Path,
) -> None:
    points_xy = _to_numpy_points(points_world)
    grid_size = int(result.get("grid_size", 256))
    point_radius_px = int(result.get("point_radius_px", 2))
    close_kernel = int(result.get("close_kernel", 5))
    open_kernel = int(result.get("open_kernel", 3))

    # Rebuild with the same parameters from result.
    origin = np.asarray(result.get("origin_xy", [points_xy[:, 0].min(), points_xy[:, 1].min()]), dtype=np.float32)
    scale = float(result.get("scale", 1.0))
    pix = np.round((points_xy - origin) * scale).astype(np.int32)
    pix = np.clip(pix, 0, grid_size - 1)
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if point_radius_px <= 0:
        mask[pix[:, 1], pix[:, 0]] = 255
    else:
        for x, y in pix:
            cv2.circle(mask, (int(x), int(y)), int(point_radius_px), 255, -1)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = _largest_component(mask)

    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tri_pts = result.get("triangle_xy_pixel", None)
    if tri_pts is not None:
        tri = np.asarray(tri_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [tri], isClosed=True, color=(0, 165, 255), thickness=2)

    success = bool(result.get("success", False))
    txt = (
        f"success={int(success)} "
        f"iou={float(result.get('iou_tri', 0.0)):.3f} "
        f"cover={float(result.get('cover', 0.0)):.3f} "
        f"v={int(result.get('vertex_count', 0))}"
    )
    cv2.putText(vis, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
