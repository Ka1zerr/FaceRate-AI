"""
Face Assessment App — Umax Clone
FastAPI backend with MediaPipe FaceLandmarker for facial proportion analysis.
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
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
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


def _dist(a, b) -> float:
    """Euclidean distance between two NormalizedLandmark points."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _compute_metrics(landmarks):
    """Return a dict of facial proportion metrics (Looksmaxxing focus)."""
    lm = landmarks

    def _dist_y(a, b): return abs(a.y - b.y)

    face_height = _dist(lm[FOREHEAD], lm[CHIN])
    bizygomatic_width = _dist(lm[LEFT_CHEEK], lm[RIGHT_CHEEK])
    bigonial_width = _dist(lm[LEFT_JAW], lm[RIGHT_JAW])
    
    # 1. FWHR
    midface_height = _dist_y(lm[BROW_CENTER], lm[UPPER_LIP])
    fwhr = bizygomatic_width / midface_height if midface_height else 0

    # 2. Jaw to Cheek ratio
    jaw_cheek_ratio = bigonial_width / bizygomatic_width if bizygomatic_width else 0

    # 3. Canthal Tilt
    left_dx = abs(lm[LEFT_EYE_INNER].x - lm[LEFT_EYE_OUTER].x)
    left_dy = lm[LEFT_EYE_INNER].y - lm[LEFT_EYE_OUTER].y
    left_tilt = math.degrees(math.atan2(left_dy, left_dx)) if left_dx else 0

    right_dx = abs(lm[RIGHT_EYE_INNER].x - lm[RIGHT_EYE_OUTER].x)
    right_dy = lm[RIGHT_EYE_INNER].y - lm[RIGHT_EYE_OUTER].y
    right_tilt = math.degrees(math.atan2(right_dy, right_dx)) if right_dx else 0

    canthal_tilt = (left_tilt + right_tilt) / 2

    # 4. Eye Aspect Ratio (Hunter eyes have low ratio)
    left_eye_w = _dist(lm[LEFT_EYE_INNER], lm[LEFT_EYE_OUTER])
    left_eye_h = _dist(lm[LEFT_EYE_TOP], lm[LEFT_EYE_BOTTOM])
    right_eye_w = _dist(lm[RIGHT_EYE_INNER], lm[RIGHT_EYE_OUTER])
    right_eye_h = _dist(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM])
    
    eye_aspect_ratio = ((left_eye_h / left_eye_w) + (right_eye_h / right_eye_w)) / 2 if left_eye_w and right_eye_w else 0

    # 5. Lower Third
    nose_chin = _dist(lm[NOSE_BOTTOM], lm[CHIN])
    lower_third_ratio = nose_chin / face_height if face_height else 0

    # 6. Symmetry
    left_sym = _dist(lm[NOSE_TIP], lm[LEFT_CHEEK])
    right_sym = _dist(lm[NOSE_TIP], lm[RIGHT_CHEEK])
    symmetry = min(left_sym, right_sym) / max(left_sym, right_sym) if max(left_sym, right_sym) else 1.0

    return {
        "fwhr": round(fwhr, 3),
        "jaw_cheek_ratio": round(jaw_cheek_ratio, 3),
        "canthal_tilt": round(canthal_tilt, 1),
        "eye_aspect_ratio": round(eye_aspect_ratio, 3),
        "lower_third_ratio": round(lower_third_ratio, 3),
        "symmetry": round(symmetry, 3),
    }


def _map_stat(val: float, avg: float, ideal: float, sd: float) -> float:
    """
    Map a biometric value to a 1.0 - 10.0 score based on population statistics.
    Average maps exactly to 5.0. Ideal maps exactly to 10.0.
    Standard deviation determines how fast the score drops in the negative direction,
    or penalizes extreme overshoot.
    """
    if ideal > avg:
        if val >= ideal:
            # Overshoot penalty
            overshoot = (val - ideal) / sd
            s = 10.0 - (overshoot * 2.0)
        elif val >= avg:
            # Interpolate between average and ideal
            s = 5.0 + 5.0 * ((val - avg) / (ideal - avg))
        else:
            # Below average
            undershoot = (avg - val) / sd
            s = 5.0 - (undershoot * 2.5)
    else:
        # Smaller is better
        if val <= ideal:
            # Overshoot penalty (too small)
            overshoot = (ideal - val) / sd
            s = 10.0 - (overshoot * 2.0)
        elif val <= avg:
            # Interpolate between average and ideal
            s = 5.0 + 5.0 * ((avg - val) / (avg - ideal))
        else:
            # Worse than average
            undershoot = (val - avg) / sd
            s = 5.0 - (undershoot * 2.5)
            
    return max(1.0, min(10.0, s))

def _score_from_metrics(m: dict) -> tuple[float, dict[str, dict]]:
    """Produce a strict composite score evaluating 'looksmaxxing' traits against population averages."""
    traits: dict[str, dict] = {}

    # 1. FWHR (Facial Width-to-Height Ratio) | Avg: 1.70, Ideal: 1.95, SD: 0.10
    fwhr = m["fwhr"]
    s_fwhr = _map_stat(fwhr, avg=1.70, ideal=1.95, sd=0.10)
    
    if s_fwhr >= 9.0: comment = "Модельная ширина лица (Chad)"
    elif s_fwhr >= 7.0: comment = "Хорошая ширина"
    elif s_fwhr >= 4.0: comment = "Средние пропорции"
    else: comment = "Лицо слишком узкое"
    traits["FWHR (Ширина лица)"] = {"score": round(s_fwhr, 1), "comment": comment, "value": fwhr}

    # 2. Canthal Tilt | Avg: 2.0°, Ideal: 6.5°, SD: 2.5°
    tilt = m["canthal_tilt"]
    s_tilt = _map_stat(tilt, avg=2.0, ideal=6.5, sd=2.5)
    
    if s_tilt >= 9.0: comment = "Острый позитивный наклон (Хищный взгляд)"
    elif s_tilt >= 7.0: comment = "Легкий позитивный наклон"
    elif s_tilt >= 4.0: comment = "Средний угол (Нейтральный)"
    else: comment = "Опущенные уголки (Prey eyes)"
    traits["Canthal Tilt (Глаза)"] = {"score": round(s_tilt, 1), "comment": comment, "value": tilt}

    # 3. Hunter Eyes (Eye Aspect Ratio) | Avg: 0.38, Ideal: 0.28, SD: 0.04
    ear = m["eye_aspect_ratio"]
    s_ear = _map_stat(ear, avg=0.38, ideal=0.28, sd=0.04)
    
    if s_ear >= 9.0: comment = "Настоящие Hunter Eyes"
    elif s_ear >= 7.0: comment = "Привлекательный узкий прищур"
    elif s_ear >= 4.0: comment = "Обычный разрез глаз"
    else: comment = "Слишком круглые глаза (Bug eyes)"
    traits["Hunter Eyes (Разрез)"] = {"score": round(s_ear, 1), "comment": comment, "value": ear}

    # 4. Jaw to Cheek Ratio | Avg: 0.78, Ideal: 0.90, SD: 0.05
    jcr = m["jaw_cheek_ratio"]
    s_jcr = _map_stat(jcr, avg=0.78, ideal=0.90, sd=0.05)
    
    if s_jcr >= 9.0: comment = "Мощная квадратная челюсть"
    elif s_jcr >= 7.0: comment = "Хорошие углы нижней челюсти"
    elif s_jcr >= 4.0: comment = "Обычная челюсть"
    else: comment = "Узкая челюсть (Рецессия)"
    traits["Jawline (Челюсть)"] = {"score": round(s_jcr, 1), "comment": comment, "value": jcr}

    # 5. Lower Third Ratio | Avg: 0.32, Ideal: 0.34, SD: 0.02
    lt = m["lower_third_ratio"]
    s_lt = _map_stat(lt, avg=0.32, ideal=0.34, sd=0.02)
    
    if s_lt >= 9.0: comment = "Идеальный массивный подбородок"
    elif s_lt >= 7.0: comment = "Хорошая проекция подбородка"
    elif s_lt >= 4.0: comment = "Средняя длина подбородка"
    else: comment = "Укороченный подбородок"
    traits["Нижняя треть"] = {"score": round(s_lt, 1), "comment": comment, "value": lt}

    # 6. Symmetry | Avg: 0.93, Ideal: 0.99, SD: 0.03
    sym = m["symmetry"]
    s_sym = _map_stat(sym, avg=0.93, ideal=0.99, sd=0.03)
    
    if s_sym >= 9.0: comment = "Идеальная симметрия"
    elif s_sym >= 7.0: comment = "Высокая симметрия"
    elif s_sym >= 4.0: comment = "Средняя симметрия"
    else: comment = "Заметная асимметрия"
    traits["Симметрия"] = {"score": round(s_sym, 1), "comment": comment, "value": sym}

    # Composite — strict weighted penalty logic.
    # To get an 8+, EVERYTHING must be good. Bad traits penalize heavily.
    scores = [s_fwhr, s_tilt, s_ear, s_jcr, s_lt, s_sym]
    weights = [0.25, 0.15, 0.15, 0.25, 0.10, 0.10]
    
    base_score = sum(s * w for s, w in zip(scores, weights))
    
    # Non-linear penalty: if any core features are below average (5.0), pull the whole score down.
    min_core = min(s_fwhr, s_jcr, s_ear)
    if min_core < 5.0:
        base_score -= (5.0 - min_core) * 0.5 
        
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

        # MediaPipe expects RGB numpy array wrapped in mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return JSONResponse(
                status_code=400,
                content={"error": "Лицо не обнаружено. Попробуйте другое фото."},
            )

        landmarks = result.face_landmarks[0]
        metrics = _compute_metrics(landmarks)
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

        return {
            "score": score,
            "verdict": verdict_text,
            "traits": traits_list,
            "metrics": metrics,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
