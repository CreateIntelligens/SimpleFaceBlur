from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
import tempfile
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import re
import time
import base64
import requests
import mimetypes
from dotenv import load_dotenv
from face_blur_onnx import FaceBlurToolONNX
from prompts import PROMPTS
import io

# 載入環境變數
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

face_blur = FaceBlurToolONNX()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emoji 列表
EMOJIS = ["😊", "🥰", "😄", "😃", "😁", "🤗", "😺", "😸"]

def call_gemini_cartoonize(image_path: str):
    """呼叫 Gemini API 進行人臉卡通化"""
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-image-preview")
    prompt = PROMPTS.get("cartoonize_faces", "")

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    # 讀取圖片並轉為 Base64
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    model_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_string,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        },
    }

    try:
        response = requests.post(
            model_url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        image_b64 = None
        for candidate in result.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                inline = part.get("inline_data") or part.get("inlineData")
                if inline and inline.get("data"):
                    image_b64 = inline["data"]
                    break
            if image_b64:
                break

        if not image_b64:
            raise ValueError(f"Unexpected response format from Gemini: {result}")

        image_data = base64.b64decode(image_b64)
        output_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        return cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise e

def _safe_stem(filename: Optional[str]) -> str:
    if not filename:
        return "image"
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "image"

def _build_output_path(mode: str, original_filename: Optional[str]) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(original_filename)
    safe_mode = re.sub(r"[^A-Za-z0-9._-]+", "_", mode)
    filename = f"{timestamp}_{stem}_{safe_mode}.jpg"
    return os.path.join(OUTPUT_DIR, filename)

def _gemini_cartoonize_faces(img, bboxes):
    """
    對整張圖片用 Gemini 卡通化，然後只把選中的人臉區域貼回原圖
    """
    if not bboxes:
        return img

    print(f"[DEBUG] Original image shape: {img.shape}", flush=True)

    # 把整張原圖送給 Gemini 卡通化
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    try:
        cv2.imwrite(temp_input.name, img)
        print(f"[DEBUG] Sending full image to Gemini for cartoonize...", flush=True)
        cartoon_img = call_gemini_cartoonize(temp_input.name)
    finally:
        temp_input.close()
        try:
            os.unlink(temp_input.name)
        except OSError:
            pass

    if cartoon_img is None:
        print(f"[DEBUG] Gemini API failed, returning original image", flush=True)
        return img

    print(f"[DEBUG] Gemini returned image shape: {cartoon_img.shape}", flush=True)

    # 確保尺寸一致
    if cartoon_img.shape[:2] != img.shape[:2]:
        print(f"[DEBUG] Resizing from {cartoon_img.shape[:2]} to {img.shape[:2]}", flush=True)
        cartoon_img = cv2.resize(cartoon_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 只把選中的人臉區域從卡通圖貼回原圖
    result = img.copy()
    for x1, y1, x2, y2 in bboxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # 確保座標在圖片範圍內
        h, w = img.shape[:2]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = cartoon_img[y1:y2, x1:x2]

    return result

def get_emoji_font(size):
    """取得支援 emoji 的字型"""
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    return ImageFont.load_default()

@app.get('/health')
def health():
    return {'status': 'ok'}

def save_upload_to_temp(upload: UploadFile):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    try:
        temp_input.write(upload.file.read())
    finally:
        temp_input.close()
    return temp_input.name

def draw_face_boxes(img, faces, selected_ids=None, hover_id=None):
    """在圖片上繪製人臉框"""
    img_with_boxes = img.copy()

    if selected_ids is None:
        selected_ids = set()

    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        face_id = face['id']

        # 決定顏色和粗細
        if face_id == hover_id:
            color = (0, 255, 255)  # 黃色 - 懸停
            thickness = 4
        elif face_id in selected_ids:
            color = (0, 0, 255)  # 紅色 - 選中
            thickness = 3
        else:
            color = (0, 255, 0)  # 綠色 - 未選中
            thickness = 2

        # 繪製矩形框
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

        # 繪製編號標籤
        label = f"#{face_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # 繪製文字背景
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            color,
            -1
        )

        # 繪製文字
        cv2.putText(
            img_with_boxes,
            label,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

    return img_with_boxes

@app.post('/detect')
def detect(image: UploadFile = File(None)):
    """檢測人臉，返回座標和帶框的圖片"""
    if image is None:
        return JSONResponse({'error': '未上傳圖片'}, status_code=400)

    # 儲存臨時檔案供 detect_faces 使用
    temp_input_name = save_upload_to_temp(image)

    try:
        _, faces = face_blur.detect_faces(temp_input_name)

        face_list = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            face_list.append({
                'id': face['id'],
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'confidence': float(face['confidence']),
                'area': int(face['area'])
            })

        return {'faces': face_list}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/preview')
def preview(
    image: UploadFile = File(None),
    selected_ids: str = Form('[]'),
    mode: str = Form('blur'),
    emoji: Optional[str] = Form(None)
):
    """返回帶有人臉框或 emoji 的預覽圖片"""
    if image is None:
        return JSONResponse({'error': '未上傳圖片'}, status_code=400)

    temp_input_name = save_upload_to_temp(image)

    try:
        img, faces = face_blur.detect_faces(temp_input_name)
        selected_ids_set = set(json.loads(selected_ids))

        # Preview 只顯示人臉框，不顯示 emoji 效果
        img_with_boxes = draw_face_boxes(img, faces, selected_ids_set)

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_output.name, img_with_boxes)

        return FileResponse(temp_output.name, media_type='image/jpeg')
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/blur')
def blur(
    image: UploadFile = File(None),
    faces: str = Form('[]'),
    mode: str = Form('emoji'),
    emoji: Optional[str] = Form(None)
):
    """對選中的人臉進行遮蔽處理（支援 emoji 或模糊）"""
    if image is None:
        return JSONResponse({'error': '未上傳圖片'}, status_code=400)

    print(f"[DEBUG] blur_mode received: '{mode}', emoji: '{emoji}'", flush=True)
    # print(f"[DEBUG] faces: {faces[:200]}", flush=True)

    # 儲存臨時檔案
    temp_input_name = save_upload_to_temp(image)

    try:
        # 讀取圖片
        img = cv2.imread(temp_input_name)
        selected_faces = json.loads(faces)

        print(f"[DEBUG] selected_faces count: {len(selected_faces)}", flush=True)

        if mode == 'blur':
            # 使用高斯模糊
            for face in selected_faces:
                x1, y1, x2, y2 = int(face['x1']), int(face['y1']), int(face['x2']), int(face['y2'])
                face_region = img[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                img[y1:y2, x1:x2] = blurred
        elif mode == 'cartoon':
            bboxes = [
                (int(face['x1']), int(face['y1']), int(face['x2']), int(face['y2']))
                for face in selected_faces
            ]
            if bboxes:
                print("[DEBUG] Calling Gemini API with face mask...", flush=True)
                img = _gemini_cartoonize_faces(img, bboxes)
            else:
                print("[DEBUG] No selected faces for cartoon mode.", flush=True)
        else:
            # 使用 Emoji 遮蔽 - 呼叫 FaceBlurToolONNX 的方法以使用統一的字型處理
            # 轉換 faces 格式以符合 library 預期 (bbox)
            lib_faces = []
            for f in selected_faces:
                lib_faces.append({
                    'id': f.get('id', 0),
                    'bbox': [int(f['x1']), int(f['y1']), int(f['x2']), int(f['y2'])]
                })

            img = face_blur.blur_faces_with_emoji(img, lib_faces, 0, 9999, custom_emojis=emoji if emoji else None)

        output_path = _build_output_path(mode, image.filename)
        cv2.imwrite(output_path, img)
        print(f"[DEBUG] Saved result to: {output_path}", flush=True)

        return FileResponse(output_path, media_type='image/jpeg')
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/process')
def process(
    image: UploadFile = File(None),
    mode: str = Form('emoji'),
    emoji: Optional[str] = Form(None)
):
    """一次性檢測並遮蔽所有人臉（支援 emoji 或模糊）"""
    if image is None:
        return JSONResponse({'error': '未上傳圖片'}, status_code=400)

    temp_input_name = save_upload_to_temp(image)

    try:
        img, faces = face_blur.detect_faces(temp_input_name)

        if mode == 'blur':
            # 使用高斯模糊
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                face_region = img[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                img[y1:y2, x1:x2] = blurred
        elif mode == 'cartoon':
            bboxes = []
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
            if bboxes:
                img = _gemini_cartoonize_faces(img, bboxes)
        else:
            # 使用 Emoji 遮蔽
            img = face_blur.blur_faces_with_emoji(img, faces, 0, 9999, custom_emojis=emoji if emoji else None)

        output_path = _build_output_path(mode, image.filename)
        cv2.imwrite(output_path, img)

        return FileResponse(output_path, media_type='image/jpeg')
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn

    print("API 啟動: http://0.0.0.0:8905")
    uvicorn.run(app, host='0.0.0.0', port=8905)
