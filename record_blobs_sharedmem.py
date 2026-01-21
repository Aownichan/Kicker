import time
import ctypes
from ctypes import wintypes
import csv
import cv2
import numpy as np

SHM_NAME = "Local\\OptiTrackFlex3Frames"

# Blob detection params
BRIGHT_THRESH = 180
MIN_AREA = 5
MAX_AREA = 50000

# Output files
CSV_PATH = "blobs_sharedmem.csv"
VIDEO_PATH = "debug_tracking.avi"

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

# Windows API bindings
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


def detect_blobs(gray: np.ndarray):
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray_blur, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)

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
    print("Waiting for shared memory producer... start the C++ MJPEGViewer + SharedMemory first.")
    hmap = None
    while hmap is None:
        h = OpenFileMappingW(FILE_MAP_READ, False, SHM_NAME)
        if not h:
            time.sleep(0.2)
            continue
        hmap = h

    # Map entire mapping by passing 0 length
    addr = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0)
    if not addr:
        CloseHandle(hmap)
        raise PermissionError(
            "MapViewOfFile failed. Ensure C++ mapping security is permissive and both processes use same elevation."
        )
    return hmap, addr


def main():
    hmap, addr = open_shared_memory()

    csv_f = None
    writer = None
    video_out = None

    last_frame_id = None
    w = h = None
    data_size = None

    # Toggles
    recording = True
    largest_only = False
    record_video = False

    # For fps estimate
    last_ts = None
    est_fps = 60.0  # fallback
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    print("Controls: [q] quit | [r] toggle CSV recording | [l] toggle largest-only/all-blobs | [v] toggle video recording")

    try:
        while True:
            # Read header
            header_bytes = ctypes.string_at(addr, HEADER_SIZE)
            hdr = SharedHeader.from_buffer_copy(header_bytes)

            if hdr.magic != MAGIC or hdr.version != 1:
                time.sleep(0.01)
                continue

            if w is None:
                w = int(hdr.width)
                h = int(hdr.height)
                data_size = int(hdr.data_size)

                if data_size != w * h:
                    raise RuntimeError(f"Header mismatch: data_size={data_size} but width*height={w*h}")

                # CSV open
                csv_f = open(CSV_PATH, "w", newline="", encoding="utf-8")
                writer = csv.writer(csv_f)
                writer.writerow(["frame_id", "timestamp_s", "blob_id", "cx", "cy", "area"])

                print(f"Connected: {w}x{h}, data_size={data_size}")
                print(f"CSV -> {CSV_PATH}")
                print(f"Video -> {VIDEO_PATH}")

            # Only process when a new frame arrives
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

            # Read grayscale frame
            img_ptr = addr + HEADER_SIZE
            img_bytes = ctypes.string_at(img_ptr, data_size)
            gray = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w))

            blobs, mask = detect_blobs(gray)

            # Visualize
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if largest_only and blobs:
                cx, cy, area = max(blobs, key=lambda t: t[2])
                cv2.circle(vis, (int(cx), int(cy)), 7, (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"area={int(area)}",
                    (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                for (cx, cy, area) in blobs:
                    cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"{int(area)}",
                        (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # Status overlay
            status = f"frame={int(hdr.frame_id)}  t={ts:.3f}s  blobs={len(blobs)}  mode={'largest' if largest_only else 'all'}  rec={'on' if recording else 'off'}  vid={'on' if record_video else 'off'}  thr={BRIGHT_THRESH}"
            cv2.putText(
                vis,
                status,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # CSV logging
            if recording and writer is not None:
                if largest_only:
                    if blobs:
                        cx, cy, area = max(blobs, key=lambda t: t[2])
                        writer.writerow([int(hdr.frame_id), ts, 0, float(cx), float(cy), float(area)])
                else:
                    for blob_id, (cx, cy, area) in enumerate(blobs):
                        writer.writerow([int(hdr.frame_id), ts, blob_id, float(cx), float(cy), float(area)])

            # Video recording (debug)
            if record_video:
                if video_out is None:
                    # Use estimated fps (will be a best effort)
                    video_out = cv2.VideoWriter(VIDEO_PATH, fourcc, float(est_fps), (w, h), True)
                    if not video_out.isOpened():
                        video_out = None
                        print("Warning: could not open VideoWriter. Video recording disabled.")
                        record_video = False
                if video_out is not None:
                    video_out.write(vis)

            cv2.imshow("OptiTrack (shared memory) + blobs", vis)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                print(f"CSV recording: {'ON' if recording else 'OFF'}")
            elif key == ord("l"):
                largest_only = not largest_only
                print(f"Mode: {'LARGEST ONLY' if largest_only else 'ALL BLOBS'}")
            elif key == ord("v"):
                record_video = not record_video
                print(f"Video recording: {'ON' if record_video else 'OFF'}")
                if not record_video and video_out is not None:
                    video_out.release()
                    video_out = None

    finally:
        if video_out is not None:
            video_out.release()
        if csv_f is not None:
            csv_f.close()

        cv2.destroyAllWindows()
        UnmapViewOfFile(addr)
        CloseHandle(hmap)


if __name__ == "__main__":
    main()
