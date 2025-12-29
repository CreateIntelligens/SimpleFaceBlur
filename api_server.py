from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tempfile
import json
import cv2
import numpy as np
from face_blur_onnx import FaceBlurToolONNX

app = Flask(__name__)
CORS(app)

face_blur = FaceBlurToolONNX()

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

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

@app.route('/detect', methods=['POST'])
def detect():
    """檢測人臉，返回座標和帶框的圖片"""
    if 'image' not in request.files:
        return jsonify({'error': '未上傳圖片'}), 400

    file = request.files['image']

    # 儲存臨時檔案供 detect_faces 使用
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_input.name)

    try:
        img, faces = face_blur.detect_faces(temp_input.name)

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

        return jsonify({'faces': face_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview():
    """返回帶有人臉框的預覽圖片"""
    if 'image' not in request.files:
        return jsonify({'error': '未上傳圖片'}), 400

    file = request.files['image']
    selected_ids_json = request.form.get('selected_ids', '[]')

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_input.name)

    try:
        img, faces = face_blur.detect_faces(temp_input.name)
        selected_ids = set(json.loads(selected_ids_json))

        # 繪製人臉框
        img_with_boxes = draw_face_boxes(img, faces, selected_ids)

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_output.name, img_with_boxes)

        return send_file(temp_output.name, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/blur', methods=['POST'])
def blur():
    """對選中的人臉進行模糊處理"""
    if 'image' not in request.files:
        return jsonify({'error': '未上傳圖片'}), 400

    file = request.files['image']
    faces_json = request.form.get('faces', '[]')

    # 儲存臨時檔案
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_input.name)

    try:
        # 重新檢測以獲取完整的 faces 資料
        img, all_faces = face_blur.detect_faces(temp_input.name)

        selected_faces = json.loads(faces_json)
        selected_ids = [f['id'] for f in selected_faces]

        # 使用高斯模糊來遮蔽選中的人臉
        for face in all_faces:
            if face['id'] in selected_ids:
                x1, y1, x2, y2 = face['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 提取人臉區域並模糊
                face_region = img[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                img[y1:y2, x1:x2] = blurred

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_output.name, img)

        return send_file(temp_output.name, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    """一次性檢測並模糊所有人臉"""
    if 'image' not in request.files:
        return jsonify({'error': '未上傳圖片'}), 400

    file = request.files['image']

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_input.name)

    try:
        img, faces = face_blur.detect_faces(temp_input.name)

        # 模糊所有人臉
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            face_region = img[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
            img[y1:y2, x1:x2] = blurred

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_output.name, img)

        return send_file(temp_output.name, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("API 啟動: http://0.0.0.0:8905")
    app.run(host='0.0.0.0', port=8905)
