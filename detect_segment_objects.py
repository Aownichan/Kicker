import time
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import cv2

# =========================================================
# Utility helpers
# =========================================================

def is_finite_xy(x: float, y: float) -> bool:
    return np.isfinite(x) and np.isfinite(y)

def quad_area(corners_xy: np.ndarray) -> float:
    x = corners_xy[:, 0]
    y = corners_xy[:, 1]
    return 0.5 * abs(
        x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0]
        - (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0])
    )

def corners_valid(corners: np.ndarray, min_sep_px=8.0, min_area_px2=5000.0) -> bool:
    if corners is None or corners.shape != (4, 2):
        return False
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < min_sep_px:
                return False
    if quad_area(corners) < min_area_px2:
        return False
    return True

def robust_jitter_px(samples: np.ndarray) -> float:
    med = np.median(samples, axis=0)
    d = np.linalg.norm(samples - med[None, :], axis=1)
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad)

# =========================================================
# Corner estimation (per-frame)
# =========================================================

def estimate_corners_from_anchors(
    pts_xy: List[Tuple[float, float]],
    W: int,
    H: int,
    max_anchor_dist_px: float = 450.0,
) -> Optional[np.ndarray]:
    if len(pts_xy) < 4:
        return None

    pts = np.array(pts_xy, dtype=np.float32)

    anchors = np.array(
        [
            [0.0, 0.0],              # TL
            [float(W), 0.0],         # TR
            [float(W), float(H)],    # BR
            [0.0, float(H)],         # BL
        ],
        dtype=np.float32
    )

    dmat = np.linalg.norm(anchors[:, None, :] - pts[None, :, :], axis=2)

    chosen = [-1, -1, -1, -1]
    used = set()

    pairs = []
    for ci in range(4):
        for pi in range(len(pts)):
            pairs.append((float(dmat[ci, pi]), ci, pi))
    pairs.sort(key=lambda x: x[0])

    for dist, ci, pi in pairs:
        if chosen[ci] != -1:
            continue
        if pi in used:
            continue
        if dist > max_anchor_dist_px:
            continue
        chosen[ci] = pi
        used.add(pi)
        if all(c != -1 for c in chosen):
            break

    if any(c == -1 for c in chosen):
        return None

    corners = pts[np.array(chosen, dtype=np.int32)]
    return corners

# =========================================================
# Corner locking
# =========================================================

@dataclass
class CornerLockConfig:
    collect_frames: int = 60
    max_jitter_px: float = 6.0
    max_drift_px: float = 25.0
    max_missing_frames: int = 30
    ema_alpha: float = 0.03

class CornerLocker:
    def __init__(self, cfg: CornerLockConfig):
        self.cfg = cfg
        self.locked: bool = False
        self.locked_corners: Optional[np.ndarray] = None
        self.buf = deque(maxlen=cfg.collect_frames)
        self.missing_frames = 0

    def reset(self):
        self.locked = False
        self.locked_corners = None
        self.buf.clear()
        self.missing_frames = 0

    def update(self, corners_est: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
        if corners_est is None or not corners_valid(corners_est):
            if self.locked:
                self.missing_frames += 1
                if self.missing_frames > self.cfg.max_missing_frames:
                    self.reset()
                    return None, "UNLOCKED (missing too long)"
                return self.locked_corners, f"LOCKED (missing {self.missing_frames})"
            else:
                return None, "SEARCHING (no corners)"
        else:
            self.missing_frames = 0

        if not self.locked:
            self.buf.append(corners_est)

            if len(self.buf) < self.cfg.collect_frames:
                return None, f"COLLECTING ({len(self.buf)}/{self.cfg.collect_frames})"

            stack = np.stack(list(self.buf), axis=0)
            med = np.median(stack, axis=0)

            jitters = [robust_jitter_px(stack[:, i, :]) for i in range(4)]
            worst_jitter = float(max(jitters))

            if worst_jitter <= self.cfg.max_jitter_px and corners_valid(med):
                self.locked = True
                self.locked_corners = med.astype(np.float32)
                return self.locked_corners, f"LOCKED (jitter={worst_jitter:.2f}px)"
            else:
                return None, f"COLLECTING (unstable jitter={worst_jitter:.2f}px)"
        else:
            drift = float(np.max(np.linalg.norm(corners_est - self.locked_corners, axis=1)))
            if drift > self.cfg.max_drift_px:
                self.reset()
                return None, f"UNLOCKED (drift={drift:.1f}px)"

            if self.cfg.ema_alpha > 0.0:
                a = self.cfg.ema_alpha
                self.locked_corners = ((1 - a) * self.locked_corners + a * corners_est).astype(np.float32)

            return self.locked_corners, f"LOCKED (drift={drift:.1f}px)"

# =========================================================
# Shared memory config
# =========================================================

SHM_NAME = "Local\\OptiTrackFlex3Objects"
MAGIC = 0x424F544F
VERSION = 1

MAX_MATCH_DIST_PX = 35.0
MAX_MISSES = 10
MIN_AREA = 5.0
MAX_AREA = 50000.0

CORNER_BAND_BIG_AREA = (106.0, 125.0)
CORNER_BAND_BIG_RADIUS = (5.85, 6.15)

CORNER_BAND_SMALL_AREA = (56.0, 72.0)
CORNER_BAND_SMALL_RADIUS = (4.20, 4.75)

CORNER_BORDER_PX = 220

# =========================================================
# Step 2: Rod assignment (Option B: known rod lines) - VERTICAL rods
# =========================================================

ROD_X = [
    0.13, 0.24, 0.34, 0.45,
    0.55, 0.65, 0.76, 0.87,
]

ROD_BAND_HALF_WIDTH = 0.046
NUM_RODS = 8

def assign_to_rods_by_known_x(blobs_uv, rod_x_list, band_half_width):
    rod_map = {i: [] for i in range(len(rod_x_list))}
    for b in blobs_uv:
        u = b["u"]
        d = [abs(u - rx) for rx in rod_x_list]
        ri = int(np.argmin(d))
        if d[ri] <= band_half_width:
            rod_map[ri].append(b)
    return rod_map

# =========================================================
# Windows API
# =========================================================

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

OpenFileMappingW = kernel32.OpenFileMappingW
OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
OpenFileMappingW.restype = wintypes.HANDLE

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
MapViewOfFile.restype = ctypes.c_void_p

UnmapViewOfFile = kernel32.UnmapViewOfFile
CloseHandle = kernel32.CloseHandle

FILE_MAP_READ = 0x0004

# =========================================================
# Shared memory structs
# =========================================================

class SharedHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("seq", ctypes.c_uint32),
        ("frame_id", ctypes.c_uint32),
        ("timestamp_s", ctypes.c_double),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("object_count", ctypes.c_uint32),
        ("max_objects", ctypes.c_uint32),
    ]

OBJECT_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("area", np.float32),
        ("radius", np.float32),
        ("left", np.int32),
        ("right", np.int32),
        ("top", np.int32),
        ("bottom", np.int32),
        ("pseudo_label", np.int32),
        ("flags", np.uint32),
    ],
    align=False,
)

HEADER_SIZE = ctypes.sizeof(SharedHeader)
ENTRY_SIZE = OBJECT_DTYPE.itemsize

# =========================================================
# Shared memory helpers
# =========================================================

def open_shared_memory():
    print("Waiting for shared memory producer (SegmentMode objects)...")
    while True:
        hmap = OpenFileMappingW(FILE_MAP_READ, False, SHM_NAME)
        if hmap:
            break
        time.sleep(0.2)

    addr = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0)
    if not addr:
        CloseHandle(hmap)
        raise RuntimeError("MapViewOfFile failed")

    return hmap, addr

def read_consistent_snapshot(addr):
    hdr1 = SharedHeader.from_buffer_copy(ctypes.string_at(addr, HEADER_SIZE))
    if hdr1.magic != MAGIC or hdr1.version != VERSION:
        return None, None

    if hdr1.seq & 1:
        return None, None

    n = min(hdr1.object_count, hdr1.max_objects)
    obj_ptr = addr + HEADER_SIZE
    objs = np.frombuffer(
        ctypes.string_at(obj_ptr, n * ENTRY_SIZE),
        dtype=OBJECT_DTYPE,
        count=n,
    ).copy()

    hdr2 = SharedHeader.from_buffer_copy(ctypes.string_at(addr, HEADER_SIZE))
    if hdr1.seq != hdr2.seq or (hdr2.seq & 1):
        return None, None

    return hdr2, objs

# =========================================================
# Simple tracker
# =========================================================

@dataclass
class Track:
    track_id: int
    x: float
    y: float
    area: float
    radius: float
    misses: int = 0
    age: int = 0

class SimpleTracker:
    def __init__(self, max_dist_px, max_misses):
        self.max_dist_px = max_dist_px
        self.max_misses = max_misses
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections):
        for t in self.tracks.values():
            t.age += 1

        if not detections:
            for t in self.tracks.values():
                t.misses += 1
            self._prune()
            return list(self.tracks.values())

        if not self.tracks:
            for x, y, a, r in detections:
                self.tracks[self.next_id] = Track(self.next_id, x, y, a, r)
                self.next_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_pos = np.array([(self.tracks[i].x, self.tracks[i].y) for i in track_ids])
        det_pos = np.array([(d[0], d[1]) for d in detections])

        dists = np.linalg.norm(track_pos[:, None] - det_pos[None, :], axis=2)

        assigned_t = set()
        assigned_d = set()

        while True:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            if dists[idx] > self.max_dist_px:
                break
            ti, di = idx
            if ti in assigned_t or di in assigned_d:
                dists[ti, di] = np.inf
                continue
            assigned_t.add(ti)
            assigned_d.add(di)

            tid = track_ids[ti]
            x, y, a, r = detections[di]
            tr = self.tracks[tid]
            tr.x, tr.y, tr.area, tr.radius = x, y, a, r
            tr.misses = 0

            dists[ti, :] = np.inf
            dists[:, di] = np.inf

        for i, tid in enumerate(track_ids):
            if i not in assigned_t:
                self.tracks[tid].misses += 1

        for i, det in enumerate(detections):
            if i not in assigned_d:
                x, y, a, r = det
                self.tracks[self.next_id] = Track(self.next_id, x, y, a, r)
                self.next_id += 1

        self._prune()
        return list(self.tracks.values())

    def _prune(self):
        self.tracks = {k: v for k, v in self.tracks.items() if v.misses <= self.max_misses}

# =========================================================
# Homography / normalization
# =========================================================

def build_homography(src_corners: np.ndarray) -> np.ndarray:
    src = src_corners.astype(np.float32)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def apply_homography_point(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    p = np.array([[[x, y]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(p, M)[0, 0]
    return float(uv[0]), float(uv[1])

# =========================================================
# Overlay mouse readout
# =========================================================

_MOUSE_XY = (-1, -1)
def _mouse_cb(event, x, y, flags, param):
    global _MOUSE_XY
    if event == cv2.EVENT_MOUSEMOVE:
        _MOUSE_XY = (x, y)

# =========================================================
# Main
# =========================================================

def main():
    hmap, addr = open_shared_memory()
    tracker = SimpleTracker(MAX_MATCH_DIST_PX, MAX_MISSES)

    locker = CornerLocker(CornerLockConfig(
        collect_frames=60,
        max_jitter_px=6.0,
        max_drift_px=25.0,
        max_missing_frames=30,
        ema_alpha=0.03,
    ))

    show_uv_labels = False
    print_rod_medians = False
    last_print = time.time()

    last_frame_id = None
    win_name = "SegmentMode + Corner Lock + Rod Assignment"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _mouse_cb)

    print("Controls: [q] quit | [space] reset corner lock | [d] toggle u,v labels | [p] toggle rod median prints")

    try:
        while True:
            hdr, objs = read_consistent_snapshot(addr)
            if hdr is None:
                continue
            if hdr.frame_id == last_frame_id:
                continue
            last_frame_id = hdr.frame_id

            H_img, W_img = int(hdr.height), int(hdr.width)

            detections = []
            corner_candidates = []

            for o in objs:
                x, y = float(o["x"]), float(o["y"])
                if not is_finite_xy(x, y):
                    continue

                a = float(o["area"])
                r = float(o["radius"])

                if MIN_AREA <= a <= MAX_AREA:
                    detections.append((x, y, a, r))

                near_left = x < CORNER_BORDER_PX
                near_right = x > (W_img - CORNER_BORDER_PX)
                near_top = y < CORNER_BORDER_PX
                near_bottom = y > (H_img - CORNER_BORDER_PX)
                near_any_corner = (
                    (near_left and near_top) or
                    (near_right and near_top) or
                    (near_right and near_bottom) or
                    (near_left and near_bottom)
                )

                is_big_corner = (CORNER_BAND_BIG_AREA[0] <= a <= CORNER_BAND_BIG_AREA[1]) and (CORNER_BAND_BIG_RADIUS[0] <= r <= CORNER_BAND_BIG_RADIUS[1])
                is_small_corner = (CORNER_BAND_SMALL_AREA[0] <= a <= CORNER_BAND_SMALL_AREA[1]) and (CORNER_BAND_SMALL_RADIUS[0] <= r <= CORNER_BAND_SMALL_RADIUS[1])

                if near_any_corner and (is_big_corner or is_small_corner):
                    corner_candidates.append((x, y))

            corners_est = estimate_corners_from_anchors(
                corner_candidates, W=W_img, H=H_img, max_anchor_dist_px=450.0
            )
            locked_corners, lock_status = locker.update(corners_est)

            tracks = tracker.update(detections)

            vis = np.zeros((H_img, W_img, 3), dtype=np.uint8)

            for x, y, _, _ in detections:
                cv2.circle(vis, (int(x), int(y)), 3, (140, 140, 140), 1)

            labels = ["TL", "TR", "BR", "BL"]
            if corners_est is not None:
                for pt, lab in zip(corners_est, labels):
                    cx, cy = int(pt[0]), int(pt[1])
                    cv2.circle(vis, (cx, cy), 9, (255, 255, 255), 2)
                    cv2.putText(vis, lab, (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            M = None
            if locked_corners is not None:
                for pt in locked_corners:
                    cx, cy = int(pt[0]), int(pt[1])
                    cv2.circle(vis, (cx, cy), 12, (200, 255, 200), 2)
                M = build_homography(locked_corners)

            overlay_w, overlay_h = 360, 360
            pad = 12
            ov = np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)

            rod_map = None
            blobs_uv = []

            if M is not None:
                cv2.rectangle(ov, (10, 10), (overlay_w - 10, overlay_h - 10), (255, 255, 255), 1)

                for i, rx in enumerate(ROD_X):
                    xpix = int(10 + rx * (overlay_w - 20))
                    cv2.line(ov, (xpix, 10), (xpix, overlay_h - 10), (80, 80, 80), 1)
                    cv2.putText(
                        ov, f"R{i}", (xpix + 3, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1
                    )

                for tr in tracks:
                    u, v = apply_homography_point(M, tr.x, tr.y)
                    if -0.2 <= u <= 1.2 and -0.2 <= v <= 1.2:
                        blobs_uv.append({
                            "u": u, "v": v,
                            "x": tr.x, "y": tr.y,
                            "area": tr.area, "radius": tr.radius,
                            "track_id": tr.track_id
                        })

                rod_map = assign_to_rods_by_known_x(blobs_uv, ROD_X, ROD_BAND_HALF_WIDTH)

                for b in blobs_uv:
                    u, v = b["u"], b["v"]
                    px = int(10 + u * (overlay_w - 20))
                    py = int(10 + v * (overlay_h - 20))

                    d = [abs(u - rx) for rx in ROD_X]
                    ri = int(np.argmin(d))
                    assigned = d[ri] <= ROD_BAND_HALF_WIDTH

                    if assigned:
                        cv2.circle(ov, (px, py), 2, (255, 255, 255), -1)
                    else:
                        cv2.circle(ov, (px, py), 2, (90, 90, 90), -1)

                    if show_uv_labels:
                        cv2.putText(
                            ov, f"{u:.2f},{v:.2f}",
                            (px + 3, py - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (200, 200, 200), 1
                        )

                ytxt = 16
                for i in range(NUM_RODS):
                    cnt = len(rod_map[i])
                    cv2.putText(
                        ov, f"R{i}:{cnt}", (overlay_w - 78, ytxt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
                    )
                    ytxt += 16

                mx, my = _MOUSE_XY
                if 0 <= mx < W_img and 0 <= my < H_img:
                    x0 = W_img - overlay_w - pad
                    y0 = pad
                    if x0 <= mx < x0 + overlay_w and y0 <= my < y0 + overlay_h:
                        ox = mx - x0
                        oy = my - y0
                        if 10 <= ox <= overlay_w - 10 and 10 <= oy <= overlay_h - 10:
                            u_m = (ox - 10) / float(overlay_w - 20)
                            v_m = (oy - 10) / float(overlay_h - 20)
                            cv2.putText(
                                ov, f"mouse u,v = {u_m:.3f},{v_m:.3f}",
                                (12, overlay_h - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (220, 220, 220), 1
                            )
                            cv2.drawMarker(ov, (int(ox), int(oy)), (220, 220, 220),
                                           markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

            x0 = W_img - overlay_w - pad
            y0 = pad
            if x0 >= 0 and y0 + overlay_h <= H_img:
                vis[y0:y0 + overlay_h, x0:x0 + overlay_w] = ov

            if print_rod_medians and rod_map is not None:
                now = time.time()
                if now - last_print >= 1.0:
                    last_print = now
                    meds = []
                    for i in range(NUM_RODS):
                        us = [b["u"] for b in rod_map[i]]
                        meds.append(float(np.median(us)) if len(us) > 0 else None)
                    print("rod median u:", ["-" if m is None else f"{m:.3f}" for m in meds])

            cv2.putText(vis, lock_status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(win_name, vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                locker.reset()
            if k == ord("d"):
                show_uv_labels = not show_uv_labels
            if k == ord("p"):
                print_rod_medians = not print_rod_medians

    finally:
        cv2.destroyAllWindows()
        UnmapViewOfFile(addr)
        CloseHandle(hmap)

if __name__ == "__main__":
    main()
