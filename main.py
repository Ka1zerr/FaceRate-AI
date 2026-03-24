"""
Face Assessment App — Umax Clone
FastAPI backend with MediaPipe FaceLandmarker for facial proportion analysis.
Uses 3D landmark normalization and head pose compensation for accurate
scoring regardless of camera angle, selfie lens distortion, or head pose.
"""

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

# Canonical 3D face model points (approximate metric positions in mm)
# Convention: Y-down (matches OpenCV image coordinates), Z-forward (into face)
_CANONICAL_3D_POINTS = np.array([
    [0.0,     0.0,     0.0],     # Nose tip (1)
    [0.0,    63.6,   -12.5],     # Chin (152) — below nose → positive Y
    [-43.3, -32.7,   -26.0],     # Left eye outer (33) — above nose → negative Y
    [43.3,  -32.7,   -26.0],     # Right eye outer (263) — above nose → negative Y
    [-28.9,  28.9,   -24.1],     # Left mouth corner (61) — below nose → positive Y
    [28.9,   28.9,   -24.1],     # Right mouth corner (291) — below nose → positive Y
], dtype=np.float64)

_POSE_LANDMARK_IDS = [NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH]


# ---------------------------------------------------------------------------
# 3D Head Pose Estimation
# ---------------------------------------------------------------------------
def _estimate_head_pose(landmarks, img_w: int, img_h: int):
    """
    Estimate head pose (yaw, pitch, roll) using solvePnP.
    Returns (yaw, pitch, roll) in degrees.
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

    dist_coeffs = np.zeros((4, 1))  # No lens distortion model

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

    # Sanity check: if pitch is outside ±90°, solvePnP hit the 180° ambiguity.
    # In that case, the rotation is unreliable — fall back to identity (no correction).
    if abs(pitch_deg) > 90:
        return 0.0, 0.0, 0.0, np.eye(3)

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
    """Return a dict of facial proportion metrics computed from 3D-normalized points."""

    face_height = _dist3d(pts_3d, FOREHEAD, CHIN)
    bizygomatic_width = _dist3d(pts_3d, LEFT_CHEEK, RIGHT_CHEEK)
    bigonial_width = _dist3d(pts_3d, LEFT_JAW, RIGHT_JAW)

    # 1. FWHR — Facial Width-to-Height Ratio
    midface_height = abs(pts_3d[BROW_CENTER][1] - pts_3d[UPPER_LIP][1])
    fwhr = bizygomatic_width / midface_height if midface_height > 1e-6 else 0

    # 2. Jaw to Cheek ratio
    jaw_cheek_ratio = bigonial_width / bizygomatic_width if bizygomatic_width > 1e-6 else 0

    # 3. Canthal Tilt (computed in 2D XY plane of the corrected face)
    left_dx = pts_3d[LEFT_EYE_INNER][0] - pts_3d[LEFT_EYE_OUTER][0]
    left_dy = pts_3d[LEFT_EYE_INNER][1] - pts_3d[LEFT_EYE_OUTER][1]
    left_tilt = math.degrees(math.atan2(left_dy, abs(left_dx))) if abs(left_dx) > 1e-6 else 0

    right_dx = pts_3d[RIGHT_EYE_INNER][0] - pts_3d[RIGHT_EYE_OUTER][0]
    right_dy = pts_3d[RIGHT_EYE_INNER][1] - pts_3d[RIGHT_EYE_OUTER][1]
    right_tilt = math.degrees(math.atan2(right_dy, abs(right_dx))) if abs(right_dx) > 1e-6 else 0

    canthal_tilt = (left_tilt + right_tilt) / 2

    # 4. Eye Aspect Ratio
    left_eye_w = _dist3d(pts_3d, LEFT_EYE_INNER, LEFT_EYE_OUTER)
    left_eye_h = _dist3d(pts_3d, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
    right_eye_w = _dist3d(pts_3d, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
    right_eye_h = _dist3d(pts_3d, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

    eye_aspect_ratio = 0
    if left_eye_w > 1e-6 and right_eye_w > 1e-6:
        eye_aspect_ratio = ((left_eye_h / left_eye_w) + (right_eye_h / right_eye_w)) / 2

    # 5. Lower Third
    nose_chin = _dist3d(pts_3d, NOSE_BOTTOM, CHIN)
    lower_third_ratio = nose_chin / face_height if face_height > 1e-6 else 0

    # 6. Symmetry (3D — much more robust than 2D)
    left_sym = _dist3d(pts_3d, NOSE_TIP, LEFT_CHEEK)
    right_sym = _dist3d(pts_3d, NOSE_TIP, RIGHT_CHEEK)
    symmetry = min(left_sym, right_sym) / max(left_sym, right_sym) if max(left_sym, right_sym) > 1e-6 else 1.0

    return {
        "fwhr": round(fwhr, 3),
        "jaw_cheek_ratio": round(jaw_cheek_ratio, 3),
        "canthal_tilt": round(canthal_tilt, 1),
        "eye_aspect_ratio": round(eye_aspect_ratio, 3),
        "lower_third_ratio": round(lower_third_ratio, 3),
        "symmetry": round(symmetry, 3),
    }


# ---------------------------------------------------------------------------
# Statistical Scoring
# ---------------------------------------------------------------------------
def _map_stat(val: float, avg: float, ideal: float, sd: float) -> float:
    """
    Map a biometric value to a 1.0 - 10.0 score based on population statistics.
    Average maps exactly to 5.0. Ideal maps exactly to 10.0.
    Standard deviation determines how fast the score drops.
    """
    if ideal > avg:
        if val >= ideal:
            overshoot = (val - ideal) / sd
            s = 10.0 - (overshoot * 1.5)
        elif val >= avg:
            s = 5.0 + 5.0 * ((val - avg) / (ideal - avg))
        else:
            undershoot = (avg - val) / sd
            s = 5.0 - (undershoot * 2.0)
    else:
        # Smaller is better
        if val <= ideal:
            overshoot = (ideal - val) / sd
            s = 10.0 - (overshoot * 1.5)
        elif val <= avg:
            s = 5.0 + 5.0 * ((avg - val) / (avg - ideal))
        else:
            undershoot = (val - avg) / sd
            s = 5.0 - (undershoot * 2.0)

    return max(1.0, min(10.0, s))


def _score_from_metrics(m: dict) -> tuple[float, dict[str, dict]]:
    """Produce a composite score evaluating 'looksmaxxing' traits against population averages."""
    traits: dict[str, dict] = {}

    # 1. FWHR | Avg: 1.75, Ideal: 1.95, SD: 0.12
    fwhr = m["fwhr"]
    s_fwhr = _map_stat(fwhr, avg=1.75, ideal=1.95, sd=0.12)

    if s_fwhr >= 9.0: comment = "Модельная ширина лица (Chad)"
    elif s_fwhr >= 7.0: comment = "Хорошая ширина"
    elif s_fwhr >= 4.0: comment = "Средние пропорции"
    else: comment = "Лицо слишком узкое"
    traits["FWHR (Ширина лица)"] = {"score": round(s_fwhr, 1), "comment": comment, "value": fwhr}

    # 2. Canthal Tilt | Avg: 2.5°, Ideal: 6.5°, SD: 3.0°
    tilt = m["canthal_tilt"]
    s_tilt = _map_stat(tilt, avg=2.5, ideal=6.5, sd=3.0)

    if s_tilt >= 9.0: comment = "Острый позитивный наклон (Хищный взгляд)"
    elif s_tilt >= 7.0: comment = "Легкий позитивный наклон"
    elif s_tilt >= 4.0: comment = "Средний угол (Нейтральный)"
    else: comment = "Опущенные уголки (Prey eyes)"
    traits["Canthal Tilt (Глаза)"] = {"score": round(s_tilt, 1), "comment": comment, "value": tilt}

    # 3. Hunter Eyes (Eye Aspect Ratio) | Avg: 0.36, Ideal: 0.28, SD: 0.05
    ear = m["eye_aspect_ratio"]
    s_ear = _map_stat(ear, avg=0.36, ideal=0.28, sd=0.05)

    if s_ear >= 9.0: comment = "Настоящие Hunter Eyes"
    elif s_ear >= 7.0: comment = "Привлекательный узкий прищур"
    elif s_ear >= 4.0: comment = "Обычный разрез глаз"
    else: comment = "Слишком круглые глаза (Bug eyes)"
    traits["Hunter Eyes (Разрез)"] = {"score": round(s_ear, 1), "comment": comment, "value": ear}

    # 4. Jaw to Cheek Ratio | Avg: 0.78, Ideal: 0.88, SD: 0.06
    jcr = m["jaw_cheek_ratio"]
    s_jcr = _map_stat(jcr, avg=0.78, ideal=0.88, sd=0.06)

    if s_jcr >= 9.0: comment = "Мощная квадратная челюсть"
    elif s_jcr >= 7.0: comment = "Хорошие углы нижней челюсти"
    elif s_jcr >= 4.0: comment = "Обычная челюсть"
    else: comment = "Узкая челюсть (Рецессия)"
    traits["Jawline (Челюсть)"] = {"score": round(s_jcr, 1), "comment": comment, "value": jcr}

    # 5. Lower Third Ratio | Avg: 0.33, Ideal: 0.34, SD: 0.025
    lt = m["lower_third_ratio"]
    s_lt = _map_stat(lt, avg=0.33, ideal=0.34, sd=0.025)

    if s_lt >= 9.0: comment = "Идеальный массивный подбородок"
    elif s_lt >= 7.0: comment = "Хорошая проекция подбородка"
    elif s_lt >= 4.0: comment = "Средняя длина подбородка"
    else: comment = "Укороченный подбородок"
    traits["Нижняя треть"] = {"score": round(s_lt, 1), "comment": comment, "value": lt}

    # 6. Symmetry | Avg: 0.93, Ideal: 0.99, SD: 0.035
    sym = m["symmetry"]
    s_sym = _map_stat(sym, avg=0.93, ideal=0.99, sd=0.035)

    if s_sym >= 9.0: comment = "Идеальная симметрия"
    elif s_sym >= 7.0: comment = "Высокая симметрия"
    elif s_sym >= 4.0: comment = "Средняя симметрия"
    else: comment = "Заметная асимметрия"
    traits["Симметрия"] = {"score": round(s_sym, 1), "comment": comment, "value": sym}

    # Composite — weighted average with softer penalty
    scores = [s_fwhr, s_tilt, s_ear, s_jcr, s_lt, s_sym]
    weights = [0.22, 0.15, 0.15, 0.22, 0.12, 0.14]

    base_score = sum(s * w for s, w in zip(scores, weights))

    # Softer non-linear penalty for weak core features
    min_core = min(s_fwhr, s_jcr, s_ear)
    if min_core < 5.0:
        base_score -= (5.0 - min_core) * 0.3

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
