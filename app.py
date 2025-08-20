from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import os
import base64
import io
from datetime import datetime
import uuid

import numpy as np
from PIL import Image
import pandas as pd

import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from openpyxl import load_workbook

# Classical CV imports
import cv2
from skimage.metrics import structural_similarity as ssim
import imagehash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Fixed backend image (not shown on UI)
FIXED_IMAGE_FILENAME = 'create an image.png'
FIXED_IMAGE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], FIXED_IMAGE_FILENAME)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Lazy-load CLIP
clip_model = None
clip_processor = None

# Lazy-load DINOv2
dino_model = None
dino_processor = None

def load_clip():
    global clip_model, clip_processor
    if clip_model is None:
        clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

def load_dino():
    global dino_model, dino_processor
    if dino_model is None:
        dino_model = AutoModel.from_pretrained('facebook/dinov2-base')
        dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

def cosine_to_percent_angular(cosine: float) -> float:
    angle = float(torch.arccos(torch.clamp(torch.tensor(cosine), -1.0, 1.0)))
    sim01 = 1.0 - (angle / np.pi)
    return max(0.0, min(100.0, sim01 * 100.0))


# ---------------- Semantic (CLIP/DINO) ----------------

def calculate_similarity_clip(img1_path: str, img2_bytes: bytes) -> float:
    try:
        with open(img1_path, 'rb') as f:
            if f.read() == img2_bytes:
                return 100.0
    except Exception:
        pass

    load_clip()
    image1 = Image.open(img1_path).convert('RGB')
    image2 = Image.open(io.BytesIO(img2_bytes)).convert('RGB')
    inputs = clip_processor(images=[image1, image2], return_tensors='pt')
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        f1, f2 = features[0], features[1]
        f1 = f1 / f1.norm(p=2)
        f2 = f2 / f2.norm(p=2)
        cosine = torch.matmul(f1, f2.T).item()
    return cosine_to_percent_angular(cosine)


def calculate_similarity_dino(img1_path: str, img2_bytes: bytes) -> float:
    load_dino()
    image1 = Image.open(img1_path).convert('RGB')
    image2 = Image.open(io.BytesIO(img2_bytes)).convert('RGB')
    inputs = dino_processor(images=[image1, image2], return_tensors='pt')
    with torch.no_grad():
        outputs = dino_model(**inputs)
        feats = outputs.last_hidden_state.mean(dim=1)
        f1, f2 = feats[0], feats[1]
        f1 = f1 / f1.norm(p=2)
        f2 = f2 / f2.norm(p=2)
        cosine = torch.matmul(f1, f2.T).item()
    return cosine_to_percent_angular(cosine)


# ---------------- Classical CV (SSIM/pHash/ORB) ----------------

def _load_cv2_rgb_from_path(path: str, max_side: int = 1024) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f'Failed to read image: {path}')
    h, w = img_bgr.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_cv2_rgb_from_bytes(data: bytes, max_side: int = 1024) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError('Failed to decode image bytes')
    h, w = img_bgr.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def _normalize01(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def classical_similarity_percent(img1_path: str, img2_bytes: bytes) -> float:
    img1 = _load_cv2_rgb_from_path(img1_path)
    img2 = _load_cv2_rgb_from_bytes(img2_bytes)

    # SSIM
    g1, g2 = _to_gray(img1), _to_gray(img2)
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]), interpolation=cv2.INTER_AREA)
    ssim_score, _ = ssim(g1, g2, full=True)
    ssim_sim = _normalize01(float(ssim_score), -1.0, 1.0)

    # pHash
    h1 = imagehash.phash(Image.fromarray(img1))
    h2 = imagehash.phash(Image.fromarray(img2))
    dist = int(h1 - h2)  # 0..64
    phash_sim = 1.0 - (dist / 64.0)
    phash_sim = float(max(0.0, min(1.0, phash_sim)))

    # ORB
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=5)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        orb_sim = 0.0
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            orb_sim = _normalize01(len(good), 0, 50) * 0.4
        else:
            src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or mask is None:
                orb_sim = _normalize01(len(good), 0, 200) * 0.6
            else:
                inliers = int(mask.ravel().sum())
                inlier_ratio = inliers / max(1, len(good))
                orb_sim = 0.6 * inlier_ratio + 0.4 * _normalize01(len(good), 0, 500)

    # Weighted average (visual similarity)
    overall = 0.35 * ssim_sim + 0.25 * phash_sim + 0.40 * orb_sim
    return float(max(0.0, min(100.0, overall * 100.0)))


# ---------------- Persistence ----------------

def save_to_excel(code: str, similarity: float, stored_image: str, compared_image: str, session_id: str) -> bool:
    try:
        excel_file = 'comparison_results.xlsx'
        new_data = {
            'Code': [code],
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Similarity_Score': [round(similarity, 2)],
            'Stored_Image': [stored_image],
            'Compared_Image': [compared_image],
            'Session_ID': [session_id]
        }
        try:
            df_existing = pd.read_excel(excel_file)
            df_new = pd.DataFrame(new_data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_combined = pd.DataFrame(new_data)
        df_combined.to_excel(excel_file, index=False)

        try:
            wb = load_workbook(excel_file)
            ws = wb.active
            for column_cells in ws.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells:
                    value = '' if cell.value is None else str(cell.value)
                    if len(value) > max_length:
                        max_length = len(value)
                ws.column_dimensions[column_letter].width = min(max_length + 2, 60)
            wb.save(excel_file)
        except Exception:
            pass

        return True
    except Exception as e:
        print(f'Error saving to Excel: {e}')
        return False


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/verify_code', methods=['POST'])
def verify_code():
    code = request.form.get('code', '').strip()
    if not code.isdigit():
        return jsonify({'success': False, 'message': 'Enter a number between 1 and 100.'})
    value = int(code)
    if value < 1 or value > 100:
        return jsonify({'success': False, 'message': 'Code must be between 1 and 100.'})
    return jsonify({'success': True, 'redirect': url_for('comparison_tool', code=value)})


@app.route('/comparison_tool')
def comparison_tool():
    code = request.args.get('code', '')
    if not code.isdigit():
        return redirect(url_for('login'))
    value = int(code)
    if value < 1 or value > 100:
        return redirect(url_for('login'))
    if not os.path.exists(FIXED_IMAGE_PATH):
        return f'Fixed backend image not found: {FIXED_IMAGE_PATH}', 500
    return render_template('comparison_tool.html', code=code, fixed_image_name=FIXED_IMAGE_FILENAME)


@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        data = request.get_json()
        if not data or 'imageData' not in data or 'code' not in data:
            return jsonify({'error': 'Missing image or code'}), 400
        code = str(data['code']).strip()
        if not code.isdigit() or int(code) < 1 or int(code) > 100:
            return jsonify({'error': 'Invalid code'}), 400
        model_choice = (data.get('model') or 'classical').lower()

        image_base64 = data['imageData']
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        image_bytes = base64.b64decode(image_base64)

        if not os.path.exists(FIXED_IMAGE_PATH):
            return jsonify({'error': f'Fixed backend image not found: {FIXED_IMAGE_FILENAME}'}), 500

        user_filename = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        user_filepath = os.path.join(app.config['UPLOAD_FOLDER'], user_filename)
        Image.open(io.BytesIO(image_bytes)).convert('RGB').save(user_filepath)

        if model_choice == 'clip':
            final_score = calculate_similarity_clip(FIXED_IMAGE_PATH, image_bytes)
            detail = {'clip': round(final_score, 2)}
        elif model_choice == 'dino':
            final_score = calculate_similarity_dino(FIXED_IMAGE_PATH, image_bytes)
            detail = {'dino': round(final_score, 2)}
        elif model_choice == 'ensemble':
            c = calculate_similarity_clip(FIXED_IMAGE_PATH, image_bytes)
            d = calculate_similarity_dino(FIXED_IMAGE_PATH, image_bytes)
            final_score = (c + d) / 2.0
            detail = {'clip': round(c, 2), 'dino': round(d, 2)}
        elif model_choice == 'classical':
            final_score = classical_similarity_percent(FIXED_IMAGE_PATH, image_bytes)
            detail = {'classical': round(final_score, 2)}
        else:
            # all three when available
            c = calculate_similarity_clip(FIXED_IMAGE_PATH, image_bytes)
            try:
                d = calculate_similarity_dino(FIXED_IMAGE_PATH, image_bytes)
            except Exception:
                d = c
            k = classical_similarity_percent(FIXED_IMAGE_PATH, image_bytes)
            final_score = (c + d + k) / 3.0
            detail = {'clip': round(c, 2), 'dino': round(d, 2), 'classical': round(k, 2)}

        final_score = round(float(final_score), 2)
        session_id = str(uuid.uuid4())
        save_to_excel(code, final_score, FIXED_IMAGE_FILENAME, user_filename, session_id)
        return jsonify({'similarity': final_score, 'detail': detail})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_results')
def download_results():
    code = request.args.get('code', '')
    if code != '8618':
        return jsonify({'error': 'Invalid download code'}), 403
    excel_file = 'comparison_results.xlsx'
    if not os.path.exists(excel_file):
        return jsonify({'error': 'No results file found'}), 404
    return send_from_directory('.', excel_file, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
