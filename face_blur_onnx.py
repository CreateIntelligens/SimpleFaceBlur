"""
人臉檢測和遮蔽模組 - ONNX版本（輕量化）
使用YOLO10m + ONNX Runtime代替PyTorch，體積減少90%以上
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import onnxruntime as ort
import random


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

    def blur_faces_with_emoji(self, img, faces, start_id, end_id, custom_emojis=None):
        """使用emoji遮蔽指定範圍的人臉

        Args:
            img: 原始圖片(numpy array)
            faces: 人臉檢測結果列表
            start_id: 開始遮蔽的人臉編號
            end_id: 結束遮蔽的人臉編號
            custom_emojis: 自定義 emoji 列表或單個 emoji 字符串
        """
        # 轉換為PIL圖片以便繪製emoji
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 決定使用的 emoji
        if custom_emojis:
            if isinstance(custom_emojis, str):
                emojis_to_use = [custom_emojis]
            else:
                emojis_to_use = custom_emojis
        else:
            emojis_to_use = self.emojis

        # 載入字型（嘗試使用系統emoji字型）
        font_size = 100
        try:
            # 支援不同系統的 emoji 字型
            font_paths = [
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:/Windows/Fonts/seguiemj.ttf",  # Windows
                "C:/Windows/Fonts/NotoColorEmoji.ttf",
                "C:/Windows/Fonts/seguisym.ttf"
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        print(f"[DEBUG] Loaded font from: {font_path}", flush=True)
                        break
                    except Exception as e:
                        # 嘗試 fallback 大小 (針對 Noto Color Emoji)
                        try:
                            font = ImageFont.truetype(font_path, 109)
                            print(f"[DEBUG] Loaded font from: {font_path} with fallback size 109", flush=True)
                            break
                        except:
                            print(f"[DEBUG] Failed to load font {font_path}: {e}", flush=True)
                            continue

            if font is None:
                # 如果找不到字型，使用預設字型
                print(f"[DEBUG] No font found, using default.", flush=True)
                font = ImageFont.load_default()
        except Exception as e:
            print(f"[DEBUG] Error loading font: {e}", flush=True)
            font = ImageFont.load_default()

        # 遮蔽選定範圍的人臉
        for face in faces:
            face_id = face["id"]

            # 檢查是否在遮蔽範圍內
            if start_id <= face_id <= end_id:
                x1, y1, x2, y2 = face["bbox"]
                
                print(f"[DEBUG] Drawing emoji on face #{face_id}", flush=True)

                # 計算人臉中心點
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 計算emoji大小（根據人臉大小調整）
                face_width = x2 - x1
                face_height = y2 - y1
                target_emoji_size = int(max(face_width, face_height) * 1.2)
                
                # 嘗試載入最適合大小的字型，如果失敗則使用固定大小並縮放
                current_font = font
                use_resize_method = False
                
                # 如果我們找到的是系統字型路徑，嘗試以此路徑載入正確大小
                # 但要注意 Noto Color Emoji 可能只支援特定大小 (109)
                font_path_to_use = None
                if hasattr(font, 'path') and font.path:
                    font_path_to_use = font.path
                
                if font_path_to_use:
                    try:
                        current_font = ImageFont.truetype(font_path_to_use, target_emoji_size)
                    except OSError:
                        # 可能是 bitmap font，嘗試使用標準大小 109
                        try:
                            current_font = ImageFont.truetype(font_path_to_use, 109)
                            use_resize_method = True
                        except:
                            # 如果都失敗，回退到預設
                            current_font = font

                # 隨機選擇 emoji
                emoji = random.choice(emojis_to_use)

                # 繪製
                if use_resize_method:
                    # Bitmap font 策略：畫在透明圖層上然後縮放
                    # 109 是 Noto Color Emoji 的常見大小
                    base_size = 109 
                    
                    # 建立透明圖層
                    emoji_layer = Image.new('RGBA', (base_size*2, base_size*2), (0,0,0,0))
                    emoji_draw = ImageDraw.Draw(emoji_layer)
                    
                    # 取得繪製大小
                    try:
                        bbox = emoji_draw.textbbox((0, 0), emoji, font=current_font)
                        e_w = bbox[2] - bbox[0]
                        e_h = bbox[3] - bbox[1]
                    except:
                        e_w = base_size
                        e_h = base_size
                        
                    # 居中繪製
                    draw_x = (base_size*2 - e_w) // 2
                    draw_y = (base_size*2 - e_h) // 2
                    
                    try:
                        emoji_draw.text((draw_x, draw_y), emoji, font=current_font, embedded_color=True)
                    except:
                        emoji_draw.text((draw_x, draw_y), emoji, font=current_font)
                        
                    # 裁切出 emoji 部分 (簡單處理：裁切到內容或直接縮放整層)
                    # 為了簡單，我們直接縮放圖層並貼上中心
                    
                    # 縮放到目標大小
                    scaled_layer = emoji_layer.resize((int(target_emoji_size*2), int(target_emoji_size*2)), Image.Resampling.LANCZOS)
                    
                    # 計算貼上位置
                    paste_x = center_x - int(target_emoji_size)
                    paste_y = center_y - int(target_emoji_size)
                    
                    # 貼上
                    img_pil.paste(scaled_layer, (paste_x, paste_y), scaled_layer)
                    print(f"[DEBUG] Drew resized bitmap emoji on face #{face_id}", flush=True)

                else:
                    # Vector font 策略：直接繪製
                    try:
                        bbox = draw.textbbox((0, 0), emoji, font=current_font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except:
                        text_width = target_emoji_size
                        text_height = target_emoji_size

                    text_x = center_x - text_width // 2
                    text_y = center_y - text_height // 2

                    try:
                        draw.text((text_x, text_y), emoji, font=current_font, embedded_color=True)
                        print(f"[DEBUG] Drew vector emoji on face #{face_id}", flush=True)
                    except Exception as e:
                        print(f"[DEBUG] Vector draw failed: {e}", flush=True)
                        try:
                            draw.text((text_x, text_y), emoji, font=current_font)
                        except:
                            pass

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
