"""
人臉檢測和遮蔽模組 - ONNX版本（輕量化）
使用YOLO10m + ONNX Runtime代替PyTorch，體積減少90%以上
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import onnxruntime as ort


class FaceBlurToolONNX:
    def __init__(self, model_path="Yolo10m/model.onnx"):
        """初始化人臉檢測工具（ONNX版本，使用YOLO10m）

        Args:
            model_path: ONNX模型檔案路徑
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型檔案不存在: {model_path}")

        # 載入ONNX模型
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # 使用CPU
        )

        # 獲取輸入輸出名稱
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 可愛的emoji清單
        self.emojis = ["😊", "🥰", "😄", "😃", "😁", "🤗", "😺", "😸"]

        # YOLO輸入大小
        self.input_size = 640

    def preprocess_image(self, img):
        """預處理圖片為YOLO輸入格式

        Args:
            img: OpenCV圖片

        Returns:
            處理後的圖片數據
        """
        # 調整大小
        img_resized = cv2.resize(img, (self.input_size, self.input_size))

        # BGR轉RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 歸一化到[0,1]
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # 轉換為NCHW格式 (batch, channels, height, width)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        return img_batch

    def nms(self, boxes, scores, iou_threshold=0.5):
        """Non-Maximum Suppression 過濾重複框"""
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def postprocess_output(self, output, img_shape, conf_threshold=0.25):
        """後處理YOLOv8輸出

        Args:
            output: YOLOv8模型輸出 [batch, 5, 8400]
            img_shape: 原始圖片形狀
            conf_threshold: 置信度閾值

        Returns:
            檢測到的人臉列表
        """
        # YOLOv8輸出格式：[batch, 5, 8400]
        # 需要轉置成 [8400, 5]
        # 5 = [x_center, y_center, width, height, confidence]
        predictions = output[0].T  # 轉置：[5, 8400] -> [8400, 5]

        boxes = []
        scores = []
        orig_h, orig_w = img_shape[:2]

        # 縮放比例
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        for pred in predictions:
            # 提取置信度（第5個元素，索引4）
            confidence = pred[4]

            # 跳過低置信度
            if confidence < conf_threshold:
                continue

            # 提取中心點和寬高
            x_center, y_center, width, height = pred[:4]

            # 轉換為角點座標
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # 縮放回原圖尺寸
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # 限制在圖片範圍內
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # 確保邊界框有效
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)

        # 應用 NMS 過濾重複框
        keep_indices = self.nms(boxes, scores, iou_threshold=0.5)

        faces = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes[idx]
            area = (x2 - x1) * (y2 - y1)
            faces.append({
                "bbox": [x1, y1, x2, y2],
                "area": float(area),
                "confidence": float(scores[idx])
            })

        return faces

    def detect_faces(self, image_path):
        """檢測圖片中的所有人臉

        Args:
            image_path: 圖片檔案路徑

        Returns:
            tuple: (原始圖片, 檢測結果列表)
        """
        # 讀取圖片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"無法讀取圖片: {image_path}")

        # 預處理
        input_data = self.preprocess_image(img)

        # 執行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # 後處理
        faces = self.postprocess_output(outputs[0], img.shape)

        # 按面積從大到小排序
        faces.sort(key=lambda x: x["area"], reverse=True)

        # 新增編號
        for i, face in enumerate(faces, 1):
            face["id"] = i

        return img, faces

    def draw_face_boxes(self, img, faces, selected_ids=None):
        """在圖片上繪製人臉框和編號

        Args:
            img: 原始圖片(numpy array)
            faces: 人臉檢測結果列表
            selected_ids: 選中要遮蔽的人臉ID列表

        Returns:
            numpy array: 繪製了人臉框的圖片
        """
        img_with_boxes = img.copy()

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_id = face["id"]
            area = face["area"]

            # 如果該人臉被選中，使用紅色框，否則使用綠色框
            color = (0, 0, 255) if (selected_ids and face_id in selected_ids) else (0, 255, 0)
            thickness = 3 if (selected_ids and face_id in selected_ids) else 2

            # 繪製矩形框
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

            # 繪製編號和麵積資訊
            label = f"#{face_id} ({int(area)}px²)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # 計算文字大小
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

    def blur_faces_with_emoji(self, img, faces, start_id, end_id):
        """使用emoji遮蔽指定範圍的人臉

        Args:
            img: 原始圖片(numpy array)
            faces: 人臉檢測結果列表
            start_id: 開始遮蔽的人臉編號
            end_id: 結束遮蔽的人臉編號

        Returns:
            numpy array: 遮蔽後的圖片
        """
        # 轉換為PIL圖片以便繪製emoji
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 載入字型（嘗試使用系統emoji字型）
        font_size = 100
        try:
            # Windows系統的emoji字型
            font_paths = [
                "C:/Windows/Fonts/seguiemj.ttf",  # Segoe UI Emoji
                "C:/Windows/Fonts/NotoColorEmoji.ttf",
                "C:/Windows/Fonts/seguisym.ttf"
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break

            if font is None:
                # 如果找不到字型，使用預設字型
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # 遮蔽選定範圍的人臉
        emoji_index = 0
        for face in faces:
            face_id = face["id"]

            # 檢查是否在遮蔽範圍內
            if start_id <= face_id <= end_id:
                x1, y1, x2, y2 = face["bbox"]

                # 計算人臉中心點
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 計算emoji大小（根據人臉大小調整）
                face_width = x2 - x1
                face_height = y2 - y1
                emoji_size = int(max(face_width, face_height) * 1.2)

                # 調整字型大小
                try:
                    if isinstance(font, ImageFont.FreeTypeFont):
                        emoji_font = ImageFont.truetype(font.path, emoji_size)
                    else:
                        emoji_font = font
                except Exception:
                    emoji_font = font

                # 選擇emoji
                emoji = self.emojis[emoji_index % len(self.emojis)]
                emoji_index += 1

                # 繪製emoji（計算位置使其居中）
                bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2

                draw.text((text_x, text_y), emoji, font=emoji_font, embedded_color=True)

        # 轉換回OpenCV格式
        img_with_emoji = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return img_with_emoji

    def get_face_info(self, faces):
        """獲取人臉資訊的文字描述

        Args:
            faces: 人臉檢測結果列表

        Returns:
            str: 人臉資訊描述
        """
        if not faces:
            return "未檢測到人臉"

        info = f"檢測到 {len(faces)} 個人臉（按面積從大到小排序）:\n\n"
        for face in faces:
            info += f"#{face['id']}: 面積={int(face['area'])}px², 置信度={face['confidence']:.2f}\n"

        return info


# 為了兼容性，創建別名
FaceBlurTool = FaceBlurToolONNX
