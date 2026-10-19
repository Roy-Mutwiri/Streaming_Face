"""Headless FPS benchmark using the app's own face_swapper.process_frame (same as Live mode)."""
import os, sys, time, argparse
os.environ['OMP_NUM_THREADS'] = '1'
import cv2
import torch  # loads CUDA/cuDNN DLLs bundled with torch (same order as modules/core.py)
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument('--provider', default='cuda', choices=['cuda', 'cpu'])
p.add_argument('--source', default='images/source_image.jpg')
p.add_argument('--target', default='webcam', help="'webcam' or an image/video path")
p.add_argument('--width', type=int, default=960)
p.add_argument('--height', type=int, default=540)
p.add_argument('--seconds', type=float, default=15)
p.add_argument('--enhancer', default='GPEN-256', choices=['Off', 'GPEN-256', 'GPEN-512'])
p.add_argument('--no-tensorrt', action='store_true')
p.add_argument('--save', help='write one processed frame to this path')
p.add_argument('--no-grain', action='store_true')
p.add_argument('--mask', default=None, help='face mask mode override')
args = p.parse_args()

import modules.globals as g
g.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.provider == 'cuda' else ['CPUExecutionProvider']
g.execution_threads = 5
g.many_faces = False
g.both_faces = False
g.face_index_range = -1
g.live_enhancer = args.enhancer
g.use_tensorrt = not args.no_tensorrt
g.live_mode = True  # benchmark the Live settings
g.grain_match = not args.no_grain
if args.mask: g.face_mask_mode = args.mask

from modules.face_analyser import initialize_face_analyser, get_many_faces
from modules.processors.frame import face_swapper

initialize_face_analyser()
src = cv2.imread(args.source)
source_faces = sorted(get_many_faces(src), key=lambda f: f.bbox[0])
if not source_faces:
    sys.exit('No face found in source image')
print(f'Source faces found: {len(source_faces)}')
face_swapper.get_face_swapper()
print('Swapper session providers:', face_swapper.FACE_SWAPPER.session.get_providers())

if args.target == 'webcam':
    cap = cv2.VideoCapture(g.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit('Could not open webcam')
    read = lambda: cap.read()
else:
    img = cv2.imread(args.target)
    if img is None:
        cap = cv2.VideoCapture(args.target)
        def read():
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = cap.read()
            return ok, f
    else:
        img = cv2.resize(img, (args.width, args.height))
        read = lambda: (True, img.copy())

ok, frame = read()
print(f'Frame size: {frame.shape[1]}x{frame.shape[0]}  faces in first frame: {len(get_many_faces(frame))}')

# warm-up
for _ in range(10):
    ok, frame = read()
    out = face_swapper.process_frame(source_faces, frame)
# Same three-stage pipeline as the Live preview (modules/ui.py): detect, swap and GPEN on their own threads
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from modules.face_restore import restore_faces

def detect_stage(frame):
    return frame, face_swapper.detect_frame_faces(frame)

def swap_stage(detect_future):
    frame, detected = detect_future.result()
    face_swapper.begin_deferred_restore()
    try:
        frame = face_swapper.process_frame(source_faces, frame, detected)
    finally:
        kps_list = face_swapper.end_deferred_restore()
    return frame, kps_list

def restore_stage(swap_future):
    frame, kps_list = swap_future.result()
    return restore_faces(frame, kps_list) if g.live_enhancer != 'Off' else frame

detect_worker, swap_worker, restore_worker, in_flight = ThreadPoolExecutor(1), ThreadPoolExecutor(1), ThreadPoolExecutor(1), deque()

frames, t0, win_t, win_n = 0, time.time(), time.time(), 0
while time.time() - t0 < args.seconds:
    ok, frame = read()
    if not ok:
        break
    in_flight.append(restore_worker.submit(restore_stage, swap_worker.submit(swap_stage, detect_worker.submit(detect_stage, frame))))
    if len(in_flight) < 3:
        continue
    out = in_flight.popleft().result()
    frames += 1; win_n += 1
    if time.time() - win_t >= 1:
        print(f'FPS: {win_n / (time.time() - win_t):.2f}', flush=True)
        win_t, win_n = time.time(), 0
print(f'\nAVERAGE FPS over {frames} frames: {frames / (time.time() - t0):.2f}')
if args.save:
    cv2.imwrite(args.save, out)
