import time
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
import math
import json
import os
import socketio

import numpy as np
import cv2

CALIB_FILE = "kicker_calibration.json"

# =========================================================
# Step 2: Rod assignment / rod centerline model
# =========================================================

ROD_X = [
    0.13, 0.24, 0.34, 0.45,
    0.55, 0.65, 0.76, 0.87,
]
ROD_BAND_HALF_WIDTH = 0.055
NUM_RODS = 8

# Learned rod centerline per rod: rod_u0(v) = A*v + B
ROD_LINE_A = [0.0 for _ in range(NUM_RODS)]
ROD_LINE_B = ROD_X[:]   # start from constants

def rod_u0_at(rod_idx: int, v: float) -> float:
    return ROD_LINE_A[rod_idx] * v + ROD_LINE_B[rod_idx]

# =========================================================
# Step 3: Rotation model (HEAD-only)
# =========================================================

ANGLE_MARKER_MODE = "head"
ANGLE_SIGN = -1.0  # your convention

HEAD_AREA_MAX = 999.0
FOOT_AREA_MIN = 115.0  # unused in head-mode
PLAYER_GAP_V = 0.06

# v-dependent calibration model:
# d0(v)     = D0_A*v + D0_B
# R_CW(v)   = RCW_A*v + RCW_B   (CLOCKWISE physical rotation @ ~90°)
# R_CCW(v)  = RCCW_A*v + RCCW_B (COUNTERCLOCKWISE physical rotation @ ~90°)
# CW_SIGN   = sign of d during CW90 for each rod (so we can pick model robustly)
D0_A = [0.0 for _ in range(NUM_RODS)]
D0_B = [0.0 for _ in range(NUM_RODS)]

RCW_A = [0.0 for _ in range(NUM_RODS)]
RCW_B = [0.045 for _ in range(NUM_RODS)]

RCCW_A = [0.0 for _ in range(NUM_RODS)]
RCCW_B = [0.045 for _ in range(NUM_RODS)]

CW_SIGN = [1.0 for _ in range(NUM_RODS)]  # +1 or -1 per rod

def d0_at(ri: int, v: float) -> float:
    return D0_A[ri] * v + D0_B[ri]

def rcw_at(ri: int, v: float) -> float:
    return max(1e-6, RCW_A[ri] * v + RCW_B[ri])

def rccw_at(ri: int, v: float) -> float:
    return max(1e-6, RCCW_A[ri] * v + RCCW_B[ri])

# =========================================================
# Calibration save/load
# =========================================================

def save_calibration(path: str):
    data = {
        "version": 2,
        "ROD_LINE_A": ROD_LINE_A,
        "ROD_LINE_B": ROD_LINE_B,
        "D0_A": D0_A,
        "D0_B": D0_B,
        "RCW_A": RCW_A,
        "RCW_B": RCW_B,
        "RCCW_A": RCCW_A,
        "RCCW_B": RCCW_B,
        "CW_SIGN": CW_SIGN,
        "PLAYER_GAP_V": PLAYER_GAP_V,
        "ANGLE_SIGN": ANGLE_SIGN,
        "HEAD_AREA_MAX": HEAD_AREA_MAX,
        "ROD_X": ROD_X,
        "ROD_BAND_HALF_WIDTH": ROD_BAND_HALF_WIDTH,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved calibration to {path}")

def load_calibration(path: str) -> bool:
    if not os.path.exists(path):
        print(f"ℹ️ No calibration file found at {path} (will run uncalibrated).")
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _assign(dst_list, src):
            for i in range(min(len(dst_list), len(src))):
                dst_list[i] = float(src[i])

        _assign(ROD_LINE_A, data.get("ROD_LINE_A", []))
        _assign(ROD_LINE_B, data.get("ROD_LINE_B", []))
        _assign(D0_A, data.get("D0_A", []))
        _assign(D0_B, data.get("D0_B", []))

        # Newer v2 fields:
        if "RCW_B" in data:
            _assign(RCW_A, data.get("RCW_A", []))
            _assign(RCW_B, data.get("RCW_B", []))
            _assign(RCCW_A, data.get("RCCW_A", []))
            _assign(RCCW_B, data.get("RCCW_B", []))
            _assign(CW_SIGN, data.get("CW_SIGN", []))
        else:
            # Backward compatibility: if you had RP/RN before, keep defaults
            print("ℹ️ Older calibration format detected. Loaded what we can, using defaults for CW/CCW.")

        print(f"Loaded calibration from {path}")
        return True
    except Exception as e:
        print("Failed to load calibration:", e)
        return False

# =========================================================
# Utility helpers
# =========================================================

def is_finite_xy(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y))

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

def clamp(x, a, b):
    return max(a, min(b, x))

def fit_line(samples_vy: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Fit y = a*v + b. Returns (a,b) or (None,None) if not enough samples.
    """
    if len(samples_vy) < 2:
        return None, None
    vv = np.array([s[0] for s in samples_vy], dtype=np.float32)
    yy = np.array([s[1] for s in samples_vy], dtype=np.float32)
    a, b = np.polyfit(vv, yy, 1)
    return float(a), float(b)

def sign_nonzero(x: float) -> float:
    return 1.0 if x >= 0 else -1.0

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
# Rod assignment
# =========================================================

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
# Player clustering + marker selection
# =========================================================

def cluster_players_by_v(blobs, gap=PLAYER_GAP_V):
    if not blobs:
        return []
    blobs_sorted = sorted(blobs, key=lambda b: b["v"])
    clusters = [[blobs_sorted[0]]]
    for b in blobs_sorted[1:]:
        if abs(b["v"] - clusters[-1][-1]["v"]) > gap:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    return clusters

def pick_angle_blob(cluster, rod_u0):
    if not cluster:
        return None

    if ANGLE_MARKER_MODE == "head":
        heads = [b for b in cluster if b["area"] <= HEAD_AREA_MAX]
        if not heads:
            return None
        return min(heads, key=lambda b: abs(b["u"] - rod_u0))

    foots = [b for b in cluster if b["area"] >= FOOT_AREA_MIN]
    if not foots:
        return None
    return max(foots, key=lambda b: abs(b["u"] - rod_u0))

def compute_angle_deg_from_u(u_blob, rod_u0, v, ri):
    d0v = d0_at(ri, v)
    d = (u_blob - rod_u0) - d0v

    # Choose calibration based on per-rod CW sign:
    # if d has same sign as CW_SIGN -> treat as CW side, else CCW side
    if d * CW_SIGN[ri] >= 0:
        R = rcw_at(ri, v)
    else:
        R = rccw_at(ri, v)

    x = clamp(d / max(1e-6, float(R)), -1.0, 1.0)
    ang = math.degrees(math.asin(x)) * ANGLE_SIGN
    return float(ang), float(d)

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
                self.tracks[self.next_id] = Track(self.next_id, float(x), float(y), float(a), float(r))
                self.next_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_pos = np.array([(self.tracks[i].x, self.tracks[i].y) for i in track_ids], dtype=np.float32)
        det_pos = np.array([(d[0], d[1]) for d in detections], dtype=np.float32)

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
            tr.x, tr.y, tr.area, tr.radius = float(x), float(y), float(a), float(r)
            tr.misses = 0

            dists[ti, :] = np.inf
            dists[:, di] = np.inf

        for i, tid in enumerate(track_ids):
            if i not in assigned_t:
                self.tracks[tid].misses += 1

        for i, det in enumerate(detections):
            if i not in assigned_d:
                x, y, a, r = det
                self.tracks[self.next_id] = Track(self.next_id, float(x), float(y), float(a), float(r))
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
# Calibration helpers
# =========================================================

def fit_rod_lines_from_clusters(rod_clusters):
    """
    Learn rod centerlines u = a*v + b from the current frame.
    Best done when all players are upright (0° pose).
    """
    for ri in range(NUM_RODS):
        pts = []
        for cl in rod_clusters.get(ri, []):
            v_med = float(np.median([bb["v"] for bb in cl]))
            rod_u_guess = rod_u0_at(ri, v_med)
            b = pick_angle_blob(cl, rod_u0=rod_u_guess)
            if b is None:
                continue
            pts.append((b["v"], b["u"]))  # (v, u)

        a, b0 = fit_line(pts)
        if a is not None and b0 is not None:
            ROD_LINE_A[ri] = a
            ROD_LINE_B[ri] = b0

def estimate_d0_const_from_clusters(rod_clusters_by_rod):
    """
    One-shot median d0 per rod (constant) from current frame (upright).
    """
    for ri in range(NUM_RODS):
        clusters = rod_clusters_by_rod.get(ri, [])
        ds = []
        for cl in clusters:
            v_med = float(np.median([bb["v"] for bb in cl]))
            rod_u0 = rod_u0_at(ri, v_med)
            b = pick_angle_blob(cl, rod_u0=rod_u0)
            if b is None:
                continue
            ds.append(b["u"] - rod_u0)
        if ds:
            D0_A[ri] = 0.0
            D0_B[ri] = float(np.median(ds))

def estimate_R_const_for_pose(rod_clusters_by_rod, pose: str):
    """
    Snapshot constant R for a physical pose:
      pose="cw"  -> sets RCW_B and CW_SIGN
      pose="ccw" -> sets RCCW_B
    Uses current d0(v).
    """
    for ri in range(NUM_RODS):
        clusters = rod_clusters_by_rod.get(ri, [])
        dd_list = []
        mag_list = []

        for cl in clusters:
            v_med = float(np.median([bb["v"] for bb in cl]))
            rod_u0 = rod_u0_at(ri, v_med)
            b = pick_angle_blob(cl, rod_u0=rod_u0)
            if b is None:
                continue

            d0v = d0_at(ri, v_med)
            dd = (b["u"] - rod_u0) - d0v
            dd_list.append(dd)
            mag_list.append(abs(dd))

        if not mag_list:
            continue

        R = float(np.median(mag_list))
        if R <= 1e-6:
            continue

        if pose == "cw":
            RCW_A[ri] = 0.0
            RCW_B[ri] = R
            # CW_SIGN is the median sign of dd in this CW pose for this rod
            CW_SIGN[ri] = sign_nonzero(float(np.median(dd_list)))
        else:
            RCCW_A[ri] = 0.0
            RCCW_B[ri] = R

# =========================================================
# Main
# =========================================================

def main():
    load_calibration(CALIB_FILE)

    show_uv_labels = False
    print_rod_medians = False
    show_angles = True
    show_rod_average = True  # one label per rod

    # recording mode
    cal_mode = None          # None, "d0", "r_cw", "r_ccw"
    cal_active = False

    cal_d0_samples = {ri: [] for ri in range(NUM_RODS)}     # (v, raw=u-rod_u0)
    cal_rcw_samples = {ri: [] for ri in range(NUM_RODS)}    # (v, |dd|) in CW pose
    cal_rccw_samples = {ri: [] for ri in range(NUM_RODS)}   # (v, |dd|) in CCW pose
    cal_cw_sign_samples = {ri: [] for ri in range(NUM_RODS)}# dd sign samples in CW pose

    hmap, addr = open_shared_memory()
    tracker = SimpleTracker(MAX_MATCH_DIST_PX, MAX_MISSES)

    locker = CornerLocker(CornerLockConfig(
        collect_frames=60,
        max_jitter_px=6.0,
        max_drift_px=25.0,
        max_missing_frames=30,
        ema_alpha=0.03,
    ))

    last_print = time.time()
    last_frame_id = None

    win_name = "SegmentMode + Corner Lock + Rod Assignment + Angle(±90) [HEAD-only]"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _mouse_cb)

    print(
        "Controls:\n"
        "  [q] quit | [space] reset corner lock\n"
        "  [d] toggle u,v labels | [p] toggle rod median prints\n"
        "  [a] toggle angles on/off | [m] toggle rod-avg vs per-player labels\n"
        "  [w] save calibration | [r] reload calibration\n"
        "  [t] print area samples\n"
        "  [l] learn rod lines u=a*v+b (do this upright)\n"
        "  [0] SNAPSHOT d0 (upright pose)\n"
        "  [9] SNAPSHOT CW90 (sets RCW and CW_SIGN)\n"
        "  [8] SNAPSHOT CCW90 (sets RCCW)\n"
        "  [u] REC d0(v) while sliding upright (optional)\n"
        "  [o] REC CW90 R(v) while sliding (optional)\n"
        "  [i] REC CCW90 R(v) while sliding (optional)\n"
        "  [s] show current models\n"
        "\n"
        f"Settings: ANGLE_MARKER_MODE={ANGLE_MARKER_MODE}, ANGLE_SIGN={ANGLE_SIGN}, HEAD_AREA_MAX={HEAD_AREA_MAX}, PLAYER_GAP_V={PLAYER_GAP_V}\n"
    )

    # -----------------------------
    # Socket.IO producer connection
    # -----------------------------
    sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=1)

    @sio.event
    def connect():
        print("Connected to Socket.IO server")

    @sio.event
    def disconnect():
        print("Disconnected from Socket.IO server")

    try:
        sio.connect("http://127.0.0.1:5000", wait_timeout=2)
    except Exception as e:
        print("Could not connect to server.py (is it running?) ->", e)
        # Continue anyway; we'll just not emit until it connects

    STREAM_HZ = 60.0
    stream_period = 1.0 / STREAM_HZ
    last_stream_time = 0.0

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
                r_ = float(o["radius"])

                if MIN_AREA <= a <= MAX_AREA:
                    detections.append((x, y, a, r_))

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

                is_big_corner = (CORNER_BAND_BIG_AREA[0] <= a <= CORNER_BAND_BIG_AREA[1]) and (CORNER_BAND_BIG_RADIUS[0] <= r_ <= CORNER_BAND_BIG_RADIUS[1])
                is_small_corner = (CORNER_BAND_SMALL_AREA[0] <= a <= CORNER_BAND_SMALL_AREA[1]) and (CORNER_BAND_SMALL_RADIUS[0] <= r_ <= CORNER_BAND_SMALL_RADIUS[1])

                if near_any_corner and (is_big_corner or is_small_corner):
                    corner_candidates.append((x, y))

            corners_est = estimate_corners_from_anchors(corner_candidates, W=W_img, H=H_img, max_anchor_dist_px=450.0)
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
            rod_clusters = {i: [] for i in range(NUM_RODS)}
            rod_state_angles = [None for _ in range(NUM_RODS)]  # always defined every frame

            if M is not None:
                cv2.rectangle(ov, (10, 10), (overlay_w - 10, overlay_h - 10), (255, 255, 255), 1)

                for i, rx in enumerate(ROD_X):
                    xpix = int(10 + rx * (overlay_w - 20))
                    cv2.line(ov, (xpix, 10), (xpix, overlay_h - 10), (80, 80, 80), 1)

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

                    cv2.circle(ov, (px, py), 2, (255, 255, 255) if assigned else (90, 90, 90), -1)

                    if show_uv_labels:
                        cv2.putText(ov, f"{u:.2f},{v:.2f} a={b['area']:.0f}",
                                    (px + 3, py - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                for ri in range(NUM_RODS):
                    rod_clusters[ri] = cluster_players_by_v(rod_map[ri], gap=PLAYER_GAP_V)

                # --- RECORDING ---
                if cal_active and cal_mode is not None:
                    for ri in range(NUM_RODS):
                        for cl in rod_clusters.get(ri, []):
                            v_med = float(np.median([bb["v"] for bb in cl]))
                            rod_u0 = rod_u0_at(ri, v_med)
                            b = pick_angle_blob(cl, rod_u0=rod_u0)
                            if b is None:
                                continue

                            raw = (b["u"] - rod_u0)
                            dd = raw - d0_at(ri, v_med)

                            if cal_mode == "d0":
                                cal_d0_samples[ri].append((v_med, raw))
                            elif cal_mode == "r_cw":
                                cal_rcw_samples[ri].append((v_med, abs(dd)))
                                cal_cw_sign_samples[ri].append(dd)
                            elif cal_mode == "r_ccw":
                                cal_rccw_samples[ri].append((v_med, abs(dd)))

                # --- RUNTIME: angles ---
                for ri in range(NUM_RODS):
                    clusters = rod_clusters.get(ri, [])
                    if not clusters:
                        continue

                    rod_angles = []

                    for cl in clusters:
                        u_med = float(np.median([bb["u"] for bb in cl]))
                        v_med = float(np.median([bb["v"] for bb in cl]))

                        rod_u0 = rod_u0_at(ri, v_med)
                        b = pick_angle_blob(cl, rod_u0=rod_u0)
                        if b is None:
                            continue

                        ang, _ = compute_angle_deg_from_u(b["u"], rod_u0, v_med, ri)
                        rod_angles.append(float(ang))

                        if show_angles and (not show_rod_average):
                            px = int(10 + u_med * (overlay_w - 20))
                            py = int(10 + v_med * (overlay_h - 20))
                            cv2.putText(ov, f"{ang:+.0f}deg", (px + 6, py - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    if show_angles and show_rod_average and rod_angles:
                        rod_ang = float(np.median(rod_angles))

                        rod_state_angles[ri] = rod_ang

                        v_ref = 0.5
                        u_ref = rod_u0_at(ri, v_ref)
                        px = int(10 + u_ref * (overlay_w - 20))

                        y_top = 26
                        y_bottom = overlay_h - 26
                        py = y_top if (ri % 2 == 0) else y_bottom

                        px = px + int((ri - (NUM_RODS - 1) / 2.0) * 3)

                        cv2.putText(
                            ov,
                            f"R{ri}:{rod_ang:+.0f}",
                            (int(px - 16), int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1
                        )

            state = {
                "t": float(hdr.timestamp_s),
                "frame": int(hdr.frame_id),
                "rods": [
                    {"id": ri, "angleDeg": (None if rod_state_angles[ri] is None else float(rod_state_angles[ri]))}
                    for ri in range(NUM_RODS)
                ],
            }

            # --- Emit state to Socket.IO server at ~60Hz ---
            now = time.time()
            if M is not None and now - last_stream_time >= stream_period:
                last_stream_time = now
                if sio.connected:
                    try:
                        sio.emit("state", state)
                    except Exception as e:
                        # Don't crash tracking if network hiccups
                        print("Emit failed:", e)

            now = time.time()

            x0 = W_img - overlay_w - pad
            y0 = pad
            if x0 >= 0 and y0 + overlay_h <= H_img:
                vis[y0:y0 + overlay_h, x0:x0 + overlay_w] = ov

            cv2.putText(vis, lock_status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if cal_active and cal_mode is not None:
                cv2.putText(vis, f"CAL {cal_mode} (REC)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow(win_name, vis)

            # keys
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                locker.reset()

            if k == ord("w"):
                save_calibration(CALIB_FILE)
            if k == ord("r"):
                load_calibration(CALIB_FILE)

            if k == ord("d"):
                show_uv_labels = not show_uv_labels
            if k == ord("p"):
                print_rod_medians = not print_rod_medians
            if k == ord("a"):
                show_angles = not show_angles
            if k == ord("m"):
                show_rod_average = not show_rod_average
                print("show_rod_average =", show_rod_average)

            if k == ord("t"):
                if blobs_uv:
                    areas = sorted([b["area"] for b in blobs_uv])
                    print("Area samples (sorted):", [f"{a:.1f}" for a in areas[:10]], "...", [f"{a:.1f}" for a in areas[-10:]])
                    print(f"Current HEAD_AREA_MAX={HEAD_AREA_MAX:.1f}")
                else:
                    print("No blobs_uv to sample.")

            if k == ord("l"):
                fit_rod_lines_from_clusters(rod_clusters)
                print("Learned rod lines u=a*v+b")
                print("A:", [f"{a:+.4f}" for a in ROD_LINE_A])
                print("B:", [f"{b:+.4f}" for b in ROD_LINE_B])

            # --- SNAPSHOT calibrations (recommended) ---
            if k == ord("0"):
                if rod_map is None:
                    print("Cannot calibrate d0: not locked.")
                else:
                    estimate_d0_const_from_clusters(rod_clusters)
                    print("SNAPSHOT d0 (upright). D0_B:", [f"{b:+.4f}" for b in D0_B])

            if k == ord("9"):
                if rod_map is None:
                    print("Cannot calibrate CW90: not locked.")
                else:
                    estimate_R_const_for_pose(rod_clusters, pose="cw")
                    print("SNAPSHOT CW90. RCW_B:", [f"{b:.4f}" for b in RCW_B])
                    print("   CW_SIGN:", [f"{s:+.0f}" for s in CW_SIGN])

            if k == ord("8"):
                if rod_map is None:
                    print("Cannot calibrate CCW90: not locked.")
                else:
                    estimate_R_const_for_pose(rod_clusters, pose="ccw")
                    print("SNAPSHOT CCW90. RCCW_B:", [f"{b:.4f}" for b in RCCW_B])

            # --- Recording (optional improvement) ---
            if k == ord("u"):
                if not cal_active:
                    for ri in range(NUM_RODS):
                        cal_d0_samples[ri].clear()
                    cal_mode = "d0"
                    cal_active = True
                    print("START recording d0(v). Slide rods while UPRIGHT. Press 'u' again to STOP+FIT.")
                elif cal_mode == "d0":
                    cal_active = False
                    for ri in range(NUM_RODS):
                        a, b = fit_line(cal_d0_samples[ri])
                        if a is not None and b is not None:
                            D0_A[ri], D0_B[ri] = a, b
                    cal_mode = None
                    print("STOP d0(v). Fitted D0_A/D0_B.")
                else:
                    print("Another recording active. Stop it first.")

            # record CW90 R(v)
            if k == ord("o"):
                if not cal_active:
                    for ri in range(NUM_RODS):
                        cal_rcw_samples[ri].clear()
                        cal_cw_sign_samples[ri].clear()
                    cal_mode = "r_cw"
                    cal_active = True
                    print("START recording CW90 R(v). Keep CW ~90° and slide rods. Press 'o' again to STOP+FIT.")
                elif cal_mode == "r_cw":
                    cal_active = False
                    for ri in range(NUM_RODS):
                        a, b = fit_line(cal_rcw_samples[ri])
                        if a is not None and b is not None:
                            RCW_A[ri], RCW_B[ri] = a, b
                        # update CW_SIGN from this recording too
                        if cal_cw_sign_samples[ri]:
                            CW_SIGN[ri] = sign_nonzero(float(np.median(cal_cw_sign_samples[ri])))
                    cal_mode = None
                    print("STOP CW90 R(v). Fitted RCW_A/RCW_B and updated CW_SIGN.")
                    print("   CW_SIGN:", [f"{s:+.0f}" for s in CW_SIGN])
                else:
                    print("Another recording active. Stop it first.")

            # record CCW90 R(v)
            if k == ord("i"):
                if not cal_active:
                    for ri in range(NUM_RODS):
                        cal_rccw_samples[ri].clear()
                    cal_mode = "r_ccw"
                    cal_active = True
                    print("START recording CCW90 R(v). Keep CCW ~90° and slide rods. Press 'i' again to STOP+FIT.")
                elif cal_mode == "r_ccw":
                    cal_active = False
                    for ri in range(NUM_RODS):
                        a, b = fit_line(cal_rccw_samples[ri])
                        if a is not None and b is not None:
                            RCCW_A[ri], RCCW_B[ri] = a, b
                    cal_mode = None
                    print("STOP CCW90 R(v). Fitted RCCW_A/RCCW_B.")
                else:
                    print("Another recording active. Stop it first.")

            if k == ord("s"):
                print("=== Rod Lines u0(v)=A*v+B ===")
                print("A:", [f"{a:+.5f}" for a in ROD_LINE_A])
                print("B:", [f"{b:+.5f}" for b in ROD_LINE_B])
                print("=== d0(v)=A*v+B ===")
                print("A:", [f"{a:+.5f}" for a in D0_A])
                print("B:", [f"{b:+.5f}" for b in D0_B])
                print("=== RCW(v)=A*v+B ===")
                print("A:", [f"{a:+.5f}" for a in RCW_A])
                print("B:", [f"{b:+.5f}" for b in RCW_B])
                print("=== RCCW(v)=A*v+B ===")
                print("A:", [f"{a:+.5f}" for a in RCCW_A])
                print("B:", [f"{b:+.5f}" for b in RCCW_B])
                print("=== CW_SIGN ===")
                print("S:", [f"{s:+.0f}" for s in CW_SIGN])

    finally:
        cv2.destroyAllWindows()
        try:
            if addr:
                UnmapViewOfFile(ctypes.c_void_p(addr))
        except Exception as e:
            print("UnmapViewOfFile failed:", e)
        try:
            if hmap:
                CloseHandle(hmap)
        except Exception as e:
            print("CloseHandle failed:", e)
        try:
            if sio.connected:
                sio.disconnect()
        except Exception:
            pass    

if __name__ == "__main__":
    main()
