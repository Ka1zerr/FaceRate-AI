"""
Face Assessment App — Umax Clone
FastAPI backend with MediaPipe FaceLandmarker for facial proportion analysis.
Uses 3D landmark normalization and head pose compensation for accurate
scoring regardless of camera angle, selfie lens distortion, or head pose.
"""

import base64
import io
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI(title="FaceRate AI")

# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker (Tasks API)
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
)

landmarker = FaceLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# Key landmark indices (478-point mesh)
# ---------------------------------------------------------------------------
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_JAW = 132
RIGHT_JAW = 361
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
BROW_CENTER = 9
UPPER_LIP = 164
NOSE_BOTTOM = 2
NOSE_TIP = 1
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# Additional landmarks for improved metrics (v3)
BROW_INNER_LEFT = 107
BROW_INNER_RIGHT = 336
BROW_OUTER_LEFT = 70
BROW_OUTER_RIGHT = 300
NOSE_WING_LEFT = 129
NOSE_WING_RIGHT = 358
# Multi-point eye aperture (Soukupová & Čech EAR)
LEFT_EYE_UPPER_1 = 160   # upper lid, near outer
LEFT_EYE_UPPER_2 = 159   # upper lid, center
LEFT_EYE_UPPER_3 = 158   # upper lid, near inner
LEFT_EYE_LOWER_1 = 144   # lower lid, near outer
LEFT_EYE_LOWER_2 = 145   # lower lid, center
LEFT_EYE_LOWER_3 = 153   # lower lid, near inner
RIGHT_EYE_UPPER_1 = 387
RIGHT_EYE_UPPER_2 = 386
RIGHT_EYE_UPPER_3 = 385
RIGHT_EYE_LOWER_1 = 373
RIGHT_EYE_LOWER_2 = 374
RIGHT_EYE_LOWER_3 = 380
# Lip landmarks
UPPER_LIP_VERMILION = 0   # top of upper lip (cupid's bow)
LOWER_LIP_VERMILION = 17  # bottom of lower lip
LIP_UPPER_INNER = 13      # inner edge of upper lip
LIP_LOWER_INNER = 14      # inner edge of lower lip

# ---------------------------------------------------------------------------
# Face contour landmark index sequences for visualization
# ---------------------------------------------------------------------------
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
             159, 160, 161, 246, 33]

_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385,
              386, 387, 388, 466, 263]

_LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

_NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]
_NOSE_BOTTOM = [98, 240, 64, 48, 115, 220, 45, 4, 275, 440, 344, 278, 294, 460, 327]

_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
               415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

# Canonical 3D face model points (approximate metric positions in mm)
# Convention: Y-down (image coordinates), Z-away from camera (into face)
# Nose tip is the most protruding point (Z=0), everything else is behind it (Z>0)
_CANONICAL_3D_POINTS = np.array([
    [0.0,     0.0,     0.0],     # Nose tip (1) — most protruding
    [0.0,    63.6,    12.5],     # Chin (152) — below nose, behind nose
    [-43.3, -32.7,    26.0],     # Left eye outer (33) — above nose, behind nose
    [43.3,  -32.7,    26.0],     # Right eye outer (263) — above nose, behind nose
    [-28.9,  28.9,    24.1],     # Left mouth corner (61) — below nose, behind nose
    [28.9,   28.9,    24.1],     # Right mouth corner (291) — below nose, behind nose
], dtype=np.float64)

_POSE_LANDMARK_IDS = [NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH]


# ---------------------------------------------------------------------------
# 3D Head Pose Estimation
# ---------------------------------------------------------------------------
def _estimate_head_pose(landmarks, img_w: int, img_h: int):
    """
    Estimate head pose (yaw, pitch, roll) using solvePnP.
    Returns (yaw, pitch, roll) in degrees + rotation matrix.
    """
    # Extract 2D image points for the 6 canonical landmarks
    image_points = np.array([
        [landmarks[idx].x * img_w, landmarks[idx].y * img_h]
        for idx in _POSE_LANDMARK_IDS
    ], dtype=np.float64)

    # Approximate camera intrinsics (focal length ~ image width)
    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        _CANONICAL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0, np.eye(3)

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Decompose rotation matrix to Euler angles
    sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
        roll = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])
    else:
        pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
        roll = 0.0

    yaw_deg = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)

    # Sanity check: if any angle is outside ±90°, solvePnP hit a 180° ambiguity.
    # Fall back to identity (no pose correction) to avoid flipping the face.
    if abs(pitch_deg) > 90 or abs(roll_deg) > 90:
        return yaw_deg, pitch_deg, roll_deg, np.eye(3)

    return yaw_deg, pitch_deg, roll_deg, rotation_mat



# ---------------------------------------------------------------------------
# 3D Landmark Normalization — project to frontal plane
# ---------------------------------------------------------------------------
def _normalize_landmarks_3d(landmarks, img_w: int, img_h: int):
    """
    Extract 3D landmark coordinates and rotate to frontal plane,
    compensating for head pose (yaw, pitch, roll).

    Returns:
      - normalized_pts: np.ndarray of shape (478, 3) — pose-corrected 3D points
      - yaw, pitch, roll: head pose angles in degrees
    """
    # Step 1: Extract raw 3D points (x, y normalized to [0,1], z is depth)
    pts_3d = np.array([
        [lm.x * img_w, lm.y * img_h, lm.z * img_w]  # z scaled by img_w (MediaPipe convention)
        for lm in landmarks
    ], dtype=np.float64)

    # Step 2: Estimate head pose
    yaw, pitch, roll, rotation_mat = _estimate_head_pose(landmarks, img_w, img_h)

    # Step 3: Center points on nose tip, apply inverse rotation, then un-center
    nose = pts_3d[NOSE_TIP].copy()
    centered = pts_3d - nose

    # Inverse rotation to "un-rotate" the face to frontal
    inv_rot = rotation_mat.T
    corrected = (inv_rot @ centered.T).T

    # Re-center on face center
    corrected = corrected + nose

    return corrected, yaw, pitch, roll


# ---------------------------------------------------------------------------
# Metric Computation — 3D-aware
# ---------------------------------------------------------------------------
def _dist3d(pts, a: int, b: int) -> float:
    """Euclidean distance between two 3D points by landmark index."""
    return float(np.linalg.norm(pts[a] - pts[b]))


def _compute_metrics(pts_3d: np.ndarray):
    """Return a dict of facial proportion metrics computed from 3D-normalized points (v3)."""

    face_height = _dist3d(pts_3d, FOREHEAD, CHIN)
    bizygomatic_width = _dist3d(pts_3d, LEFT_CHEEK, RIGHT_CHEEK)
    bigonial_width = _dist3d(pts_3d, LEFT_JAW, RIGHT_JAW)

    # 1. FWHR — stabilized with averaged brow Y from 3 points
    brow_y = float(np.mean([
        pts_3d[BROW_INNER_LEFT][1],
        pts_3d[BROW_CENTER][1],
        pts_3d[BROW_INNER_RIGHT][1],
    ]))
    lip_y = pts_3d[UPPER_LIP][1]
    midface_height = abs(lip_y - brow_y)
    fwhr = bizygomatic_width / midface_height if midface_height > 1e-6 else 0

    # 2. Jaw to Cheek ratio
    jaw_cheek_ratio = bigonial_width / bizygomatic_width if bizygomatic_width > 1e-6 else 0

    # 3. Canthal Tilt (in XY plane of corrected face)
    def _eye_tilt(outer_idx, inner_idx):
        dx = pts_3d[inner_idx][0] - pts_3d[outer_idx][0]
        dy = pts_3d[inner_idx][1] - pts_3d[outer_idx][1]
        return math.degrees(math.atan2(dy, abs(dx))) if abs(dx) > 1e-6 else 0.0

    canthal_tilt = (_eye_tilt(LEFT_EYE_OUTER, LEFT_EYE_INNER)
                    + _eye_tilt(RIGHT_EYE_OUTER, RIGHT_EYE_INNER)) / 2.0

    # 4. Eye Aspect Ratio — 3-pair EAR (Soukupová & Čech, more robust)
    def _ear_3pt(outer, inner, uppers, lowers):
        w = float(np.linalg.norm(pts_3d[outer] - pts_3d[inner]))
        if w < 1e-6:
            return 0.0
        verts = [float(np.linalg.norm(pts_3d[u] - pts_3d[l]))
                 for u, l in zip(uppers, lowers)]
        return float(np.mean(verts)) / w

    left_ear = _ear_3pt(
        LEFT_EYE_OUTER, LEFT_EYE_INNER,
        [LEFT_EYE_UPPER_1, LEFT_EYE_UPPER_2, LEFT_EYE_UPPER_3],
        [LEFT_EYE_LOWER_1, LEFT_EYE_LOWER_2, LEFT_EYE_LOWER_3],
    )
    right_ear = _ear_3pt(
        RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
        [RIGHT_EYE_UPPER_1, RIGHT_EYE_UPPER_2, RIGHT_EYE_UPPER_3],
        [RIGHT_EYE_LOWER_1, RIGHT_EYE_LOWER_2, RIGHT_EYE_LOWER_3],
    )
    eye_aspect_ratio = (left_ear + right_ear) / 2.0

    # 5. Lower Third (Chin-to-Philtrum Ratio)
    # Replaced full face height (cannot be measured accurately without hairline)
    # with the standard Chin vs Philtrum ratio. Ideal is ~2.25 (chin is 2x+ longer).
    philtrum = abs(pts_3d[NOSE_BOTTOM][1] - pts_3d[UPPER_LIP][1])
    chin_len = abs(pts_3d[LOWER_LIP_VERMILION][1] - pts_3d[CHIN][1])
    lower_third_ratio = chin_len / philtrum if philtrum > 1e-6 else 0
    # 6. Symmetry — 7 bilateral landmark pairs (much more robust)
    _sym_pairs = [
        (LEFT_EYE_OUTER, RIGHT_EYE_OUTER),
        (LEFT_EYE_INNER, RIGHT_EYE_INNER),
        (LEFT_CHEEK, RIGHT_CHEEK),
        (LEFT_JAW, RIGHT_JAW),
        (LEFT_MOUTH, RIGHT_MOUTH),
        (BROW_OUTER_LEFT, BROW_OUTER_RIGHT),
        (NOSE_WING_LEFT, NOSE_WING_RIGHT),
    ]
    midline = pts_3d[NOSE_TIP]
    sym_ratios = []
    for l_idx, r_idx in _sym_pairs:
        d_l = float(np.linalg.norm(pts_3d[l_idx] - midline))
        d_r = float(np.linalg.norm(pts_3d[r_idx] - midline))
        if max(d_l, d_r) > 1e-6:
            sym_ratios.append(min(d_l, d_r) / max(d_l, d_r))
    symmetry = float(np.mean(sym_ratios)) if sym_ratios else 1.0

    # 7. NEW — Facial Thirds Balance (0..1, 1 = perfect)
    # NOTE: Only compare middle and lower thirds — the upper third
    # (hairline to brow) cannot be measured reliably from face mesh
    # because landmark 10 is NOT the hairline, it's top of the mesh.
    middle_third = abs(brow_y - pts_3d[NOSE_BOTTOM][1])
    lower_third = abs(pts_3d[NOSE_BOTTOM][1] - pts_3d[CHIN][1])
    ml_total = middle_third + lower_third
    if ml_total > 1e-6:
        mid_ratio = middle_third / ml_total  # ideal ≈ 0.50
        thirds_balance = max(0.0, 1.0 - abs(mid_ratio - 0.50) * 4)
    else:
        thirds_balance = 0.5

    # 8. NEW — Nose Width Ratio (nose width / interocular distance)
    nose_width = _dist3d(pts_3d, NOSE_WING_LEFT, NOSE_WING_RIGHT)
    interocular = _dist3d(pts_3d, LEFT_EYE_INNER, RIGHT_EYE_INNER)
    nose_width_ratio = nose_width / interocular if interocular > 1e-6 else 0

    # 9. NEW — Lip Fullness (total lip height / lower third height)
    lip_upper_h = _dist3d(pts_3d, UPPER_LIP_VERMILION, LIP_UPPER_INNER)
    lip_lower_h = _dist3d(pts_3d, LIP_LOWER_INNER, LOWER_LIP_VERMILION)
    lip_fullness = (lip_upper_h + lip_lower_h) / lower_third if lower_third > 1e-6 else 0

    return {
        "fwhr": round(fwhr, 3),
        "jaw_cheek_ratio": round(jaw_cheek_ratio, 3),
        "canthal_tilt": round(canthal_tilt, 1),
        "eye_aspect_ratio": round(eye_aspect_ratio, 3),
        "lower_third_ratio": round(lower_third_ratio, 3),
        "symmetry": round(symmetry, 3),
        "thirds_balance": round(thirds_balance, 3),
        "nose_width_ratio": round(nose_width_ratio, 3),
        "lip_fullness": round(lip_fullness, 3),
    }


# ---------------------------------------------------------------------------
# Landmark Visualization
# ---------------------------------------------------------------------------
def _draw_landmarks_on_image(img_np: np.ndarray, landmarks, metrics: dict) -> str:
    """
    Draw face mesh, contours, and measurement lines on the image.
    Returns base64-encoded JPEG string.
    """
    canvas = img_np.copy()
    h, w = canvas.shape[:2]

    # Helper: landmark -> pixel coords
    def lm2px(idx):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    # Scale factor for line thickness based on image size
    sf = max(1, int(min(w, h) / 500))

    # --- 1. Draw all 478 mesh dots (subtle) ---
    for i in range(len(landmarks)):
        px, py = lm2px(i)
        cv2.circle(canvas, (px, py), max(1, sf // 2), (0, 220, 220), -1)

    # --- 2. Draw contours ---
    def draw_contour(indices, color, thickness=None):
        t = thickness or max(1, sf)
        pts = [lm2px(i) for i in indices]
        for j in range(len(pts) - 1):
            cv2.line(canvas, pts[j], pts[j + 1], color, t, cv2.LINE_AA)

    # Face oval — white
    draw_contour(_FACE_OVAL, (255, 255, 255), max(1, sf))
    # Eyes — green
    draw_contour(_LEFT_EYE, (0, 255, 170), max(1, sf + 1))
    draw_contour(_RIGHT_EYE, (0, 255, 170), max(1, sf + 1))
    # Eyebrows — yellow
    draw_contour(_LEFT_EYEBROW, (0, 230, 255), max(1, sf))
    draw_contour(_RIGHT_EYEBROW, (0, 230, 255), max(1, sf))
    # Nose — light blue
    draw_contour(_NOSE_BRIDGE, (255, 200, 100), max(1, sf))
    draw_contour(_NOSE_BOTTOM, (255, 200, 100), max(1, sf))
    # Lips — magenta/pink
    draw_contour(_LIPS_OUTER, (180, 100, 255), max(1, sf + 1))
    draw_contour(_LIPS_INNER, (180, 100, 255), max(1, sf))

    # --- 3. Draw measurement lines ---
    line_color = (0, 180, 255)  # orange in BGR
    label_color = (0, 180, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, sf * 0.3)
    t_line = max(1, sf)

    def draw_measurement(idx_a, idx_b, label, offset=(0, -8)):
        pa, pb = lm2px(idx_a), lm2px(idx_b)
        cv2.line(canvas, pa, pb, line_color, t_line, cv2.LINE_AA)
        # Small markers at endpoints
        cv2.circle(canvas, pa, max(2, sf + 1), line_color, -1)
        cv2.circle(canvas, pb, max(2, sf + 1), line_color, -1)
        # Label at midpoint
        mx = (pa[0] + pb[0]) // 2 + offset[0]
        my = (pa[1] + pb[1]) // 2 + offset[1]
        cv2.putText(canvas, label, (mx, my), font, font_scale, (0, 0, 0), max(1, sf + 2), cv2.LINE_AA)
        cv2.putText(canvas, label, (mx, my), font, font_scale, label_color, max(1, sf), cv2.LINE_AA)

    # FWHR — bizygomatic width
    draw_measurement(LEFT_CHEEK, RIGHT_CHEEK,
                     f"FWHR {metrics['fwhr']}", offset=(0, -10 * sf))

    # Jaw width
    draw_measurement(LEFT_JAW, RIGHT_JAW,
                     f"Jaw {metrics['jaw_cheek_ratio']}", offset=(0, 8 * sf))

    # Face height (forehead to chin)
    draw_measurement(FOREHEAD, CHIN,
                     f"H", offset=(8 * sf, 0))

    # Canthal tilt — eye corners
    draw_measurement(LEFT_EYE_OUTER, LEFT_EYE_INNER,
                     f"{metrics['canthal_tilt']}\xb0", offset=(0, -6 * sf))
    draw_measurement(RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
                     f"{metrics['canthal_tilt']}\xb0", offset=(0, -6 * sf))

    # Nose width
    draw_measurement(NOSE_WING_LEFT, NOSE_WING_RIGHT,
                     f"Nose {metrics['nose_width_ratio']}", offset=(0, 6 * sf))

    # Lip height
    draw_measurement(UPPER_LIP_VERMILION, LOWER_LIP_VERMILION,
                     f"Lip {metrics['lip_fullness']}", offset=(8 * sf, 0))

    # --- 4. Encode to base64 JPEG ---
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', canvas_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode('ascii')
    return b64


# ---------------------------------------------------------------------------
# Asymmetric Gaussian Scoring (v4)
# ---------------------------------------------------------------------------
def _score_gauss(val: float, ideal: float, sigma_lo: float, sigma_hi: float) -> float:
    """
    Score a metric value on a 1-10 scale using an asymmetric Gaussian
    centered on the ideal value.

    - Exactly at ideal -> 10.0
    - sigma_lo: tolerance for values BELOW ideal (tighter = harsher)
    - sigma_hi: tolerance for values ABOVE ideal (wider = more forgiving)
    """
    sigma = sigma_hi if val >= ideal else sigma_lo
    z = (val - ideal) / sigma
    raw = math.exp(-0.5 * z * z)
    return round(max(1.0, 1.0 + 9.0 * raw), 1)


# Metric definitions: (ideal, sigma_below, sigma_above)
_METRIC_PARAMS = {
    # Relaxed FWHR & Jaw to not destroy "pretty boy / TikTok" aesthetics
    "fwhr":              (  1.90,  0.22,  0.50),  # ideal slightly lower, much more forgiving to narrow faces
    "canthal_tilt":      (  7.0,   3.5,   5.0 ),  # positive tilt good
    "eye_aspect_ratio":  (  0.27,  0.08,  0.06),  # narrower better
    "jaw_cheek_ratio":   (  0.84,  0.08,  0.15),  # slightly narrower ideal, forgiving to diamond jawlines
    "lower_third_ratio": (  2.50,  0.40,  2.50),  # chin-to-philtrum ratio
    "symmetry":          (  1.00,  0.04,  0.10),
    "thirds_balance":    (  1.00,  0.15,  0.10),
    "nose_width_ratio":  (  1.00,  0.15,  0.25),
    "lip_fullness":      (  0.35,  0.08,  0.12),
}

_TRAIT_INFO = {
    "fwhr": {
        "name": "FWHR (Ширина лица)",
        9: "Модельная ширина лица",
        7: "Хорошая ширина",
        4: "Средние пропорции",
        0: "Узкое лицо",
    },
    "canthal_tilt": {
        "name": "Canthal Tilt (Наклон глаз)",
        9: "Хищный позитивный наклон",
        7: "Привлекательный позитивный наклон",
        4: "Нейтральный угол",
        0: "Опущенные уголки (Prey eyes)",
    },
    "eye_aspect_ratio": {
        "name": "Hunter Eyes (Разрез)",
        9: "Настоящие Hunter Eyes",
        7: "Узкий привлекательный прищур",
        4: "Обычный разрез глаз",
        0: "Круглые глаза (Bug eyes)",
    },
    "jaw_cheek_ratio": {
        "name": "Jawline (Челюсть)",
        9: "Мощная квадратная челюсть",
        7: "Хорошие углы челюсти",
        4: "Обычная челюсть",
        0: "Узкая челюсть (Рецессия)",
    },
    "lower_third_ratio": {
        "name": "Нижняя треть",
        9: "Идеальная проекция подбородка",
        7: "Хорошая длина подбородка",
        4: "Средний подбородок",
        0: "Отклонение от идеала",
    },
    "symmetry": {
        "name": "Симметрия",
        9: "Идеальная симметрия",
        7: "Высокая симметрия",
        4: "Средняя симметрия",
        0: "Заметная асимметрия",
    },
    "thirds_balance": {
        "name": "Баланс третей",
        9: "Идеальный баланс третей",
        7: "Хорошие пропорции третей",
        4: "Средний баланс",
        0: "Непропорциональные трети",
    },
    "nose_width_ratio": {
        "name": "Нос (Ширина)",
        9: "Модельно пропорциональный нос",
        7: "Аккуратная ширина носа",
        4: "Средняя ширина носа",
        0: "Широкий нос",
    },
    "lip_fullness": {
        "name": "Губы (Полнота)",
        9: "Выразительные полные губы",
        7: "Привлекательная полнота",
        4: "Средние губы",
        0: "Тонкие губы",
    },
}


def _get_comment(info: dict, score: float) -> str:
    if score >= 9.0: return info[9]
    if score >= 7.0: return info[7]
    if score >= 4.0: return info[4]
    return info[0]


def _score_from_metrics(m: dict) -> tuple[float, dict[str, dict]]:
    """Produce a composite score with 9 Gaussian-scored metrics (v4)."""
    traits: dict[str, dict] = {}
    scores: list[float] = []

    metric_keys = [
        "fwhr", "canthal_tilt", "eye_aspect_ratio", "jaw_cheek_ratio",
        "lower_third_ratio", "symmetry", "thirds_balance",
        "nose_width_ratio", "lip_fullness",
    ]
    weights = [0.16, 0.12, 0.12, 0.16, 0.08, 0.12, 0.08, 0.08, 0.08]

    for key in metric_keys:
        ideal, s_lo, s_hi = _METRIC_PARAMS[key]
        val = m[key]
        s = _score_gauss(val, ideal, s_lo, s_hi)
        scores.append(s)

        info = _TRAIT_INFO[key]
        traits[info["name"]] = {
            "score": s,
            "comment": _get_comment(info, s),
            "value": val,
        }

    # Composite — weighted average
    base_score = sum(s * w for s, w in zip(scores, weights))

    # Penalty if any core trait (FWHR, jaw, eyes) is extremely weak
    min_core = min(scores[0], scores[2], scores[3])
    if min_core < 4.0:
        base_score -= (4.0 - min_core) * 0.15

    # Consistency bonus — all traits above 7
    if min(scores) >= 7.0:
        base_score += 0.3

    composite = round(max(1.0, min(10.0, base_score)), 1)
    return composite, traits


def _verdict(score: float) -> str:
    if score >= 9:
        return "Модельная внешность! 🔥"
    elif score >= 7.5:
        return "Отличные пропорции лица 💎"
    elif score >= 6:
        return "Выше среднего — привлекательные черты ✨"
    elif score >= 4.5:
        return "Средние пропорции 👌"
    elif score >= 3:
        return "Есть отклонения от идеала"
    else:
        return "Значительные отклонения от идеала"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Accept an image upload and return face assessment."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
        img_h, img_w = img_np.shape[:2]

        # MediaPipe expects RGB numpy array wrapped in mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return JSONResponse(
                status_code=400,
                content={"error": "Лицо не обнаружено. Попробуйте другое фото."},
            )

        landmarks = result.face_landmarks[0]

        # --- 3D normalization & head pose ---
        pts_3d, yaw, pitch, roll = _normalize_landmarks_3d(landmarks, img_w, img_h)

        # Quality warnings
        quality_warnings = []
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        if abs_yaw > 25:
            quality_warnings.append(
                f"Голова сильно повёрнута ({yaw:+.0f}°). Поверните лицо к камере для точной оценки."
            )
        elif abs_yaw > 15:
            quality_warnings.append(
                f"Небольшой поворот головы ({yaw:+.0f}°). Результат может немного отличаться от фронтального."
            )

        if abs_pitch > 20:
            quality_warnings.append(
                f"Голова наклонена вверх/вниз ({pitch:+.0f}°). Старайтесь смотреть прямо."
            )

        # Compute metrics on 3D-normalized landmarks
        metrics = _compute_metrics(pts_3d)
        score, traits = _score_from_metrics(metrics)
        verdict_text = _verdict(score)

        # Generate annotated image with landmarks
        annotated_b64 = _draw_landmarks_on_image(img_np, landmarks, metrics)

        traits_list = []
        for name, info in traits.items():
            traits_list.append(
                {
                    "name": name,
                    "score": info["score"],
                    "comment": info["comment"],
                }
            )

        response = {
            "score": score,
            "verdict": verdict_text,
            "traits": traits_list,
            "metrics": metrics,
            "head_pose": {
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
                "roll": round(roll, 1),
            },
            "annotated_image": annotated_b64,
        }

        if quality_warnings:
            response["quality_warnings"] = quality_warnings

        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
