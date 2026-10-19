import os
from typing import List, Dict

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, 'workflow')

file_types = [
    ('Image', ('*.png','*.jpg','*.jpeg','*.gif','*.bmp')),
    ('Video', ('*.mp4','*.mkv'))
]


source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = None
keep_audio = None
keep_frames = None
many_faces = None

nsfw_filter = None
video_encoder = None
video_quality = None
live_mirror = None
flip_y = None
flip_x = None
live_resizable = True
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
fp_ui: Dict[str, bool] = {}
camera_input_combobox = None
webcam_preview_running = False
both_faces = None
flip_faces = None
detect_face_right = None
detect_face_right_value = False
show_target_face_box = None
mouth_mask=False
mask_feather_ratio=30
mask_down_size=1.05
mask_size=1
show_mouth_mask_box=False
flip_faces_value=False
sticky_face_value=0.20
use_pseudo_face = False  # Whether to use pseudo face at all
pseudo_face_threshold = 0.20  # Minimum score to use pseudo face
max_pseudo_face_count = 30  # Maximum number of frames to use pseudo face continuously
face_tracking=False
face_tracking_value=False
target_face1_score=0.00
target_face2_score=0.00
target_face3_score=0.00
target_face4_score=0.00
target_face5_score=0.00
target_face6_score=0.00
target_face7_score=0.00
target_face8_score=0.00
target_face9_score=0.00
target_face10_score=0.00
target_face_left_embedding=None
target_face_right_embedding=None
source_face_left_embedding=None
source_face_right_embedding=None
target_face=False

embedding_weight_size = 0.60
weight_distribution_size=1.00
position_size = 0.40
old_embedding_weight  = 0.90
new_embedding_weight  = 0.10

mouth_mask_switch_preview=None
rot_range_dropdown_preview=None
face_index_dropdown_preview=None
flipX_switch_preview=None
flipY_switch_preview=None
face_rot_range=0
face_index_range=0
isRotatedClockwise=False
isRotatedCounterClockwise=False
isRotated180=False
isRotated0=False
flicker_threshold=0.7
camera_optionmenu=None
camera_var=0
camera_index=0
use_pencil_filter=False
use_ink_filter_white=False
use_ink_filter_black=False
use_black_lines=False
face_forehead_var=0.1

auto_rotate_value = True 
auto_rotate_switch_main = None
auto_rotate_switch_preview = None
rotation_check_interval = 5

detection_frequency = 1 # Default to 3 (Balanced speed/accuracy)

# --- Speed / quality ---
use_tensorrt = True          # Run models through TensorRT FP16 when CUDA is selected (see modules/accel.py)
live_enhancer = 'GPEN-256'   # Face restoration after swap: 'Off', 'GPEN-256' or 'GPEN-512'
enhancer_blend = 0.8         # 0..1, how much of the restored face is blended over the raw swap
swap_sharpen = 0.3           # 0..1, unsharp-mask amount applied to the upscaled 128px swap output

# --- Realism ---
face_mask_mode = 'Occlusion' # 'Landmarks', 'Occlusion' (hands/hair/objects stay in front), 'Parser', 'Occlusion+Parser'
tone_match = 0.5             # 0..1, how far the swapped face's skin tone moves toward the target's
swap_resolution = 128        # 128, 256 or 512: pixel boost (256 = 4 swapper passes, 512 = 16)
temporal_smoothing = True    # Live: smooth landmark jitter between frames (One-Euro filter)
live_mode = False            # set while the Live webcam preview runs
# Image/video files are not real-time, so they get the slower, highest-quality settings
file_swap_resolution = 256   # pixel boost for files (Live uses swap_resolution)
file_enhancer = 'GPEN-512'   # restorer for files when live_enhancer is not 'Off'
average_source_faces = False # Average all faces in the source image into one identity (several photos of one person)

mask_target_option = "Both" # Add this line