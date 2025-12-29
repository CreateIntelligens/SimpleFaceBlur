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
from face_blur_onnx import FaceBlurToolONNX

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

face_blur = FaceBlurToolONNX()

# 建立 output 目錄
os.makedirs('/app/output', exist_ok=True)

# Emoji 列表
EMOJIS = ["😊", "🥰", "😄", "😃", "😁", "🤗", "😺", "😸"]

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

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_output.name, img)

        return FileResponse(temp_output.name, media_type='image/jpeg')
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
        else:
            # 使用 Emoji 遮蔽
            img = face_blur.blur_faces_with_emoji(img, faces, 0, 9999, custom_emojis=emoji if emoji else None)

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_output.name, img)

        return FileResponse(temp_output.name, media_type='image/jpeg')
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn

    print("API 啟動: http://0.0.0.0:8905")
    uvicorn.run(app, host='0.0.0.0', port=8905)
