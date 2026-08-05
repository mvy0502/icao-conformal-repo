"""
batch_predict.py — BioGaze'i bir klasördeki TÜM görüntülerde çalıştırıp
her ICAO kriteri (binary) + sürekli metrikleri + kaynak klasörü (ground-truth
etiketi olarak) tek bir CSV'ye yazar.

Kullanım:
    python batch_predict.py -i <girdi_klasoru> -o predictions.csv [--limit N]

Notlar:
- Kaynak klasör adı (source_folder) TONO'da ihlal türünü verir → ground-truth.
- torch.load güvenli-yükleme yaması (weights_only=False) en başta uygulanır.
"""
import os
import sys
import csv
import argparse
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# --- torch.load yaması (eski checkpoint'ler yeni torch'ta yüklenebilsin) ---
import torch
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

import config
from detectors import detect
from landmarks import landmark
from head_pose import headpose
from face_parser import parserModel
from emotion_recognizer import emotion_detector
from image_quality import qualitychecker
from gaze_estimation import gaze_estimator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# CSV sütunları: 15 ikili ICAO kriteri + sürekli metrikler
CRITERIA = [
    "Compliant",
    "Head_without_covering",
    "Eyes_open",
    "No_sunglasses",
    "No_posterization",
    "Gaze_in_camera",
    "Neutral_expression",
    "In_focus",
    "Correct_exposure",
    "No_or_light_makeup",
    "No_pixelation",
    "Frontal_pose",
    "Correct_saturation",
    "Uniform_background",
    "Uniform_face_lighting",
]
METRICS = ["IED", "pitch", "yaw", "roll", "eyes_open_val",
           "mouth_open_val", "has_glasses"]
CSV_HEADER = ["rel_path", "source_folder", "filename", "faces_detected"] \
             + CRITERIA + METRICS + ["error"]


import signal


class ImageTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ImageTimeout("görüntü analiz zaman aşımı")


signal.signal(signal.SIGALRM, _timeout_handler)


class BioGazeRunner:
    """Tüm alt modelleri BİR KEZ yükler, sonra her görüntüyü analiz eder."""

    def __init__(self):
        self.detector = detect.FaceDetector()
        self.landmark_recognizer = landmark.LandmarkRecognizer()
        self.pose_estimator = headpose.HeadposeEstimator()
        self.face_parser = parserModel.FaceParser()
        self.emotion_recognizer = emotion_detector.EmotionDetector()
        self.quality_checker = qualitychecker.QualityChecker()
        self.gaze_model = gaze_estimator.GazeEstimator()

    def analyze(self, image_path):
        """Tek görüntü → sonuç sözlüğü (quality_analysis.py main() mantığının aynısı)."""
        row = {k: None for k in CSV_HEADER}

        faces_detected, correct_exposure = self.detector.detector_analysis(image_path)
        row["faces_detected"] = faces_detected

        # Tam olarak bir yüz yoksa: uyumlu değil, kriterler ölçülemez
        if faces_detected != config.MAX_FACES:
            row["Compliant"] = False
            return row

        rejected = False

        # Baş pozu
        pitch, yaw, roll = self.pose_estimator.get_headpose_values(image_path)
        frontal_pose = True
        if yaw < config.MIN_YAW or yaw > config.MAX_YAW:
            frontal_pose = False; rejected = True
        if pitch < config.MIN_PITCH or pitch > config.MAX_PITCH:
            frontal_pose = False; rejected = True
        if roll < config.MIN_ROLL or roll > config.MAX_ROLL:
            frontal_pose = False; rejected = True

        # Yüz ayrıştırıcı (parser)
        (has_hat, color_saturation, has_glasses, head_not_contained,
         chin_not_contained, shoulder_check, uniform_illumination,
         homogeneous_background, has_sunglasses) = self.face_parser.parser_analysis(image_path)
        if has_hat: rejected = True
        if not homogeneous_background: rejected = True
        if not shoulder_check:
            frontal_pose = False; rejected = True
        if color_saturation: rejected = True
        if chin_not_contained or head_not_contained: rejected = True

        # Landmark
        (inter_eye_distance, eyes_open, mouth_open, m_h, m_v,
         uniform_luminosity, image_height, image_width,
         has_makeup) = self.landmark_recognizer.landmark_analysis(image_path)

        eyes_open_compliant = True
        if eyes_open < config.EYES_THRESHOLD:
            eyes_open_compliant = False; rejected = True
        if mouth_open < config.MOUTH_THRESHOLD: rejected = True
        if has_glasses and has_sunglasses: rejected = True
        if has_makeup: rejected = True
        if (not uniform_illumination) or (not uniform_luminosity): rejected = True

        # Duygu / ifade
        neutral_expression = self.emotion_recognizer.check_neutral_expression(image_path)
        if not neutral_expression: rejected = True

        # Bakış
        gaze_in_camera = self.gaze_model.calculate_gaze(image_path)
        if not gaze_in_camera: rejected = True

        # Bilgisayarla görü kontrolleri
        is_posterized = self.quality_checker.is_posterized(image_path)
        is_pixelated = self.quality_checker.is_pixelated(image_path)
        out_of_focus = self.quality_checker.is_out_of_focus(image_path)
        if is_posterized: rejected = True
        if is_pixelated: rejected = True
        if out_of_focus: rejected = True

        if not correct_exposure: rejected = True

        # --- Kriter sonuçları (quality_analysis.py satır 401 ile birebir) ---
        row.update({
            "Compliant": (not rejected),
            "Head_without_covering": (not has_hat),
            "Eyes_open": eyes_open_compliant,
            "No_sunglasses": (not (has_glasses and has_sunglasses)),
            "No_posterization": (not is_posterized),
            "Gaze_in_camera": bool(gaze_in_camera),
            "Neutral_expression": bool(neutral_expression),
            "In_focus": (not out_of_focus),
            "Correct_exposure": bool(correct_exposure),
            "No_or_light_makeup": (not has_makeup),
            "No_pixelation": (not is_pixelated),
            "Frontal_pose": frontal_pose,
            "Correct_saturation": (not color_saturation),
            "Uniform_background": bool(homogeneous_background),
            "Uniform_face_lighting": bool(uniform_illumination and uniform_luminosity),
            # sürekli metrikler
            "IED": round(float(inter_eye_distance), 4),
            "pitch": round(float(pitch), 4),
            "yaw": round(float(yaw), 4),
            "roll": round(float(roll), 4),
            "eyes_open_val": round(float(eyes_open), 6),
            "mouth_open_val": round(float(mouth_open), 6),
            "has_glasses": int(bool(has_glasses)),
        })
        return row


def collect_images(root):
    if os.path.isfile(root):
        return [root]
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser(description="BioGaze toplu tahmin -> CSV")
    ap.add_argument("-i", "--input", required=True, help="Görüntü klasörü veya dosyası")
    ap.add_argument("-o", "--output", default="predictions.csv", help="Çıktı CSV")
    ap.add_argument("--limit", type=int, default=None, help="En fazla N görüntü işle")
    ap.add_argument("--progress", type=int, default=25, help="Her N görüntüde ilerleme yaz")
    args = ap.parse_args()

    images = collect_images(args.input)
    import random
    random.Random(42).shuffle(images)  # karışık sıra: kısmi sonuç bile tüm klasörleri/kriterleri temsil etsin
    if args.limit:
        images = images[:args.limit]
    total = len(images)
    print(f"[batch] {total} görüntü bulundu. Modeller yükleniyor...", flush=True)

    runner = BioGazeRunner()
    print("[batch] Modeller hazır. Analiz başlıyor...", flush=True)

    n_ok = n_err = 0
    input_root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)

    # --- Resume: daha önce işlenmiş rel_path'leri atla (aksilikte kayıp olmasın) ---
    done = set()
    mode = "w"
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        with open(args.output) as rf:
            for r in csv.DictReader(rf):
                if r.get("rel_path"):
                    done.add(r["rel_path"])
        mode = "a"
        print(f"[batch] Devam modu: {len(done)} görüntü zaten işlenmiş, atlanıyor.", flush=True)

    with open(args.output, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if mode == "w":
            writer.writeheader()
        for idx, path in enumerate(images, 1):
            rel = os.path.relpath(path, input_root) if os.path.isdir(args.input) else os.path.basename(path)
            if rel in done:
                continue
            source_folder = os.path.basename(os.path.dirname(path))
            base = {k: None for k in CSV_HEADER}
            base["rel_path"] = rel
            base["source_folder"] = source_folder
            base["filename"] = os.path.basename(path)
            try:
                signal.alarm(60)  # görüntü başına 60 sn üst sınır: takılan görüntüyü atla
                res = runner.analyze(path)
                signal.alarm(0)
                res["rel_path"] = rel
                res["source_folder"] = source_folder
                res["filename"] = os.path.basename(path)
                writer.writerow(res)
                n_ok += 1
            except Exception as e:
                signal.alarm(0)
                base["error"] = f"{type(e).__name__}: {e}"
                writer.writerow(base)
                n_err += 1
            if idx % args.progress == 0 or idx == total:
                print(f"[batch] {idx}/{total} işlendi (ok={n_ok}, hata={n_err})", flush=True)
            f.flush()

    print(f"[batch] BİTTİ → {args.output} | başarılı={n_ok}, hata={n_err}", flush=True)


if __name__ == "__main__":
    main()
