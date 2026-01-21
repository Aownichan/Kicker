import time
import ctypes
from ctypes import wintypes
import csv
import cv2
import numpy as np

SHM_NAME = "Local\\OptiTrackFlex3Frames"

# Segment mode dot detection (blueish markers on black)
B_MIN = 120
R_MAX = 140
G_MAX = 140
MIN_AREA = 8
MAX_AREA = 20000

CSV_PATH = "segments_blobs.csv"
RAW_VIDEO_PATH = "segment_raw.avi"
DEBUG_VIDEO_PATH = "segment_debug.avi"

class SharedHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("frame_id", ctypes.c_uint32),
        ("timestamp_s", ctypes.c_double),
        ("data_size", ctypes.c_uint32),
    ]

HEADER_SIZE = ctypes.sizeof(SharedHeader)
MAGIC = 0x4853544F  # 'OTSH'

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

OpenFileMappingW = kernel32.OpenFileMappingW
OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
OpenFileMappingW.restype = wintypes.HANDLE

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
MapViewOfFile.restype = ctypes.c_void_p

UnmapViewOfFile = kernel32.UnmapViewOfFile
UnmapViewOfFile.argtypes = [ctypes.c_void_p]
UnmapViewOfFile.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

FILE_MAP_READ = 0x0004


def detect_dots_rgba(rgba: np.ndarray):
    # rgba: HxWx4, channels: R,G,B,A
    r = rgba[:, :, 0]
    g = rgba[:, :, 1]
    b = rgba[:, :, 2]

    mask = (b >= B_MIN) & (r <= R_MAX) & (g <= G_MAX)
    mask = (mask.astype(np.uint8) * 255)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        blobs.append((cx, cy, area))

    return blobs, mask


def open_shared_memory():
    print("Waiting for shared memory producer (SegmentMode)...")
    hmap = None
    while hmap is None:
        h = OpenFileMappingW(FILE_MAP_READ, False, SHM_NAME)
        if not h:
            time.sleep(0.2)
            continue
        hmap = h

    addr = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0)
    if not addr:
        CloseHandle(hmap)
        raise PermissionError("MapViewOfFile failed. Check C++ mapping security / elevation.")
    return hmap, addr


def main():
    hmap, addr = open_shared_memory()

    csv_f = None
    writer = None
    raw_out = None
    dbg_out = None

    last_frame_id = None
    w = h = None
    data_size = None

    recording_csv = False
    record_raw = False
    record_debug = False
    largest_only = False

    last_ts = None
    est_fps = 60.0
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    print("Controls: [q] quit | [r] CSV | [1] raw video | [2] debug video | [l] largest/all")

    try:
        while True:
            hdr_bytes = ctypes.string_at(addr, HEADER_SIZE)
            hdr = SharedHeader.from_buffer_copy(hdr_bytes)

            if hdr.magic != MAGIC or hdr.version != 1:
                time.sleep(0.01)
                continue

            if w is None:
                w = int(hdr.width)
                h = int(hdr.height)
                data_size = int(hdr.data_size)

                expected = w * h * 4
                if data_size != expected:
                    raise RuntimeError(f"Expected data_size={expected} (w*h*4) but got {data_size}")

                csv_f = open(CSV_PATH, "w", newline="", encoding="utf-8")
                writer = csv.writer(csv_f)
                writer.writerow(["frame_id", "timestamp_s", "blob_id", "cx", "cy", "area"])

                print(f"Connected: {w}x{h}, RGBA bytes={data_size}")
                print(f"CSV -> {CSV_PATH}")
                print(f"Raw -> {RAW_VIDEO_PATH}")
                print(f"Debug -> {DEBUG_VIDEO_PATH}")

            if last_frame_id == hdr.frame_id:
                time.sleep(0.001)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue
            last_frame_id = hdr.frame_id

            ts = float(hdr.timestamp_s)
            if last_ts is not None:
                dt = ts - last_ts
                if dt > 1e-6:
                    est_fps = max(1.0, min(240.0, 1.0 / dt))
            last_ts = ts

            img_ptr = addr + HEADER_SIZE
            img_bytes = ctypes.string_at(img_ptr, data_size)
            rgba = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, 4))

            blobs, mask = detect_dots_rgba(rgba)

            # Convert for display/recording
            vis = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

            if largest_only and blobs:
                cx, cy, area = max(blobs, key=lambda t: t[2])
                cv2.circle(vis, (int(cx), int(cy)), 7, (0, 255, 0), 2)
            else:
                for cx, cy, area in blobs:
                    cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)

            cv2.putText(
                vis,
                f"frame={int(hdr.frame_id)} t={ts:.3f}s fps~{est_fps:.1f} blobs={len(blobs)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if recording_csv:
                if largest_only:
                    if blobs:
                        cx, cy, area = max(blobs, key=lambda t: t[2])
                        writer.writerow([int(hdr.frame_id), ts, 0, float(cx), float(cy), float(area)])
                else:
                    for blob_id, (cx, cy, area) in enumerate(blobs):
                        writer.writerow([int(hdr.frame_id), ts, blob_id, float(cx), float(cy), float(area)])

            if record_raw and raw_out is None:
                raw_out = cv2.VideoWriter(RAW_VIDEO_PATH, fourcc, float(est_fps), (w, h), True)
                if not raw_out.isOpened():
                    print("Warning: raw VideoWriter failed; disabling raw recording.")
                    raw_out = None
                    record_raw = False

            if record_debug and dbg_out is None:
                dbg_out = cv2.VideoWriter(DEBUG_VIDEO_PATH, fourcc, float(est_fps), (w, h), True)
                if not dbg_out.isOpened():
                    print("Warning: debug VideoWriter failed; disabling debug recording.")
                    dbg_out = None
                    record_debug = False

            if record_raw and raw_out is not None:
                raw_out.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))

            if record_debug and dbg_out is not None:
                dbg_out.write(vis)

            cv2.imshow("Segmentmode RGBA and detections", vis)
            cv2.imshow("Dot mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording_csv = not recording_csv
                print(f"CSV: {'ON' if recording_csv else 'OFF'}")
            elif key == ord("1"):
                record_raw = not record_raw
                print(f"Raw video: {'ON' if record_raw else 'OFF'}")
                if not record_raw and raw_out is not None:
                    raw_out.release()
                    raw_out = None
            elif key == ord("2"):
                record_debug = not record_debug
                print(f"Debug video: {'ON' if record_debug else 'OFF'}")
                if not record_debug and dbg_out is not None:
                    dbg_out.release()
                    dbg_out = None
            elif key == ord("l"):
                largest_only = not largest_only
                print(f"Mode: {'LARGEST' if largest_only else 'ALL'}")

    finally:
        if raw_out is not None:
            raw_out.release()
        if dbg_out is not None:
            dbg_out.release()
        if csv_f is not None:
            csv_f.close()

        cv2.destroyAllWindows()
        UnmapViewOfFile(addr)
        CloseHandle(hmap)


if __name__ == "__main__":
    main()
