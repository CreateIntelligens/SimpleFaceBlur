"""
GUI介面 - 人臉遮蔽工具（現代化版本）
使用CustomTkinter建立現代化圖形使用者介面
使用YOLO10m + ONNX 實現輕量化臉部偵測
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import os
import numpy as np
from face_blur_onnx import FaceBlurTool

# 設定外觀模式和顏色主題
ctk.set_appearance_mode("light")  # 可選: "light", "dark", "system"
ctk.set_default_color_theme("blue")  # 可選: "blue", "dark-blue", "green"


class ModernFaceBlurGUI:
    def __init__(self, root):
        """初始化現代化GUI介面

        Args:
            root: CustomTkinter根視窗
        """
        self.root = root
        self.root.title("人臉遮蔽工具 - Face Blur Tool")
        self.root.geometry("1400x900")

        # 初始化人臉檢測工具
        try:
            self.blur_tool = FaceBlurTool()
        except FileNotFoundError as e:
            messagebox.showerror("錯誤", str(e))
            self.root.destroy()
            return

        # 目前狀態
        self.current_image = None
        self.current_faces = []
        self.current_image_path = None
        self.preview_image = None

        # 新增：互動式選擇狀態
        self.selected_face_ids = set()      # 選中要遮蔽的人臉ID
        self.current_tool = "pen"           # "pen" 或 "eraser"
        self.hover_face_id = None           # 懸停的人臉ID

        # 新增：坐標轉換參數
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Canvas 圖片對象
        self.canvas_image_id = None
        self.tk_image = None

        # 建立UI元件
        self.create_widgets()

    def create_widgets(self):
        """建立所有UI元件"""

        # ===== 頂部標題列 =====
        title_frame = ctk.CTkFrame(self.root, height=80, corner_radius=0)
        title_frame.pack(side="top", fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)

        title_label = ctk.CTkLabel(
            title_frame,
            text="😊 人臉遮蔽工具",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(side="left", padx=30, pady=20)

        # 主題切換按鈕
        self.theme_switch = ctk.CTkSwitch(
            title_frame,
            text="深色模式",
            command=self.toggle_theme,
            font=ctk.CTkFont(size=13)
        )
        self.theme_switch.pack(side="right", padx=30)

        # ===== 主容器 =====
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(side="top", fill="both", expand=True, padx=20, pady=20)

        # ===== 左側：圖片顯示區域 =====
        left_frame = ctk.CTkFrame(main_container, corner_radius=15)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # 圖片區域標題
        img_title_frame = ctk.CTkFrame(left_frame, fg_color="transparent", height=60)
        img_title_frame.pack(fill="x", padx=20, pady=(20, 10))
        img_title_frame.pack_propagate(False)

        ctk.CTkLabel(
            img_title_frame,
            text="圖片預覽",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left", pady=10)

        # 上傳和檢測按鈕放在右側
        btn_container = ctk.CTkFrame(img_title_frame, fg_color="transparent")
        btn_container.pack(side="right")

        self.upload_btn = ctk.CTkButton(
            btn_container,
            text="📁 上傳圖片",
            command=self.upload_image,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=140,
            corner_radius=10
        )
        self.upload_btn.pack(side="left", padx=5)

        self.detect_btn = ctk.CTkButton(
            btn_container,
            text="🔍 檢測人臉",
            command=self.detect_faces,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=140,
            corner_radius=10,
            fg_color="#2563eb",
            hover_color="#1d4ed8"
        )
        self.detect_btn.pack(side="left", padx=5)

        # ===== 工具欄 =====
        toolbar_frame = ctk.CTkFrame(left_frame, height=60, corner_radius=10)
        toolbar_frame.pack(fill="x", padx=20, pady=(0, 10))
        toolbar_frame.pack_propagate(False)

        # 工具欄標籤
        ctk.CTkLabel(
            toolbar_frame,
            text="選擇工具：",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=(15, 10))

        # 筆工具按鈕
        self.pen_btn = ctk.CTkButton(
            toolbar_frame,
            text="🖊️ 筆（選擇）",
            command=lambda: self.select_tool("pen"),
            font=ctk.CTkFont(size=13, weight="bold"),
            width=130,
            height=40,
            corner_radius=8,
            fg_color="#ec4899",
            hover_color="#db2777"
        )
        self.pen_btn.pack(side="left", padx=5)

        # 橡皮擦工具按鈕
        self.eraser_btn = ctk.CTkButton(
            toolbar_frame,
            text="🧹 橡皮擦（取消）",
            command=lambda: self.select_tool("eraser"),
            font=ctk.CTkFont(size=13, weight="bold"),
            width=150,
            height=40,
            corner_radius=8,
            fg_color="#6b7280",
            hover_color="#4b5563"
        )
        self.eraser_btn.pack(side="left", padx=5)

        # 全選按鈕
        self.select_all_btn = ctk.CTkButton(
            toolbar_frame,
            text="✅ 全選",
            command=self.select_all_faces,
            font=ctk.CTkFont(size=13),
            width=100,
            height=40,
            corner_radius=8
        )
        self.select_all_btn.pack(side="left", padx=5)

        # 全不選按鈕
        self.deselect_all_btn = ctk.CTkButton(
            toolbar_frame,
            text="❌ 全不選",
            command=self.deselect_all_faces,
            font=ctk.CTkFont(size=13),
            width=100,
            height=40,
            corner_radius=8
        )
        self.deselect_all_btn.pack(side="left", padx=5)

        # 圖片顯示區域（帶陰影效果）
        img_display_frame = ctk.CTkFrame(left_frame, corner_radius=10)
        img_display_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # 替換為 Canvas（支援互動）
        self.canvas = tk.Canvas(
            img_display_frame,
            bg="#2b2b2b",  # 深色背景
            highlightthickness=0,
            cursor="hand2"
        )
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # 綁定事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_hover)
        self.canvas.bind("<Leave>", self.on_canvas_leave)

        # ===== 右側：控制面板 =====
        right_frame = ctk.CTkFrame(main_container, width=380, corner_radius=15)
        right_frame.pack(side="right", fill="y", padx=(10, 0))
        right_frame.pack_propagate(False)

        # 檢測結果區域
        result_title = ctk.CTkLabel(
            right_frame,
            text="檢測結果",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        result_title.pack(pady=(25, 15), padx=20)

        # 人臉資訊文字框（使用ScrollableFrame）
        info_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        info_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        self.face_info_text = ctk.CTkTextbox(
            info_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8,
            wrap="word"
        )
        self.face_info_text.pack(fill="both", expand=True, padx=3, pady=3)
        self.face_info_text.insert("1.0", "請上傳圖片並點選檢測人臉")

        # 選擇狀態區域
        settings_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        settings_frame.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(
            settings_frame,
            text="選擇狀態",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        # 選擇狀態標籤
        self.selection_status_label = ctk.CTkLabel(
            settings_frame,
            text="尚未檢測人臉",
            font=ctk.CTkFont(size=13),
            wraplength=320
        )
        self.selection_status_label.pack(pady=10, padx=15)

        # 查看選擇按鈕
        self.view_selection_btn = ctk.CTkButton(
            settings_frame,
            text="👁️ 查看選擇",
            command=self.view_selection,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            corner_radius=8,
            fg_color="#f59e0b",
            hover_color="#d97706"
        )
        self.view_selection_btn.pack(fill="x", padx=15, pady=(10, 15))

        # 執行遮蔽按鈕
        self.blur_btn = ctk.CTkButton(
            right_frame,
            text="😊 執行遮蔽",
            command=self.apply_blur,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45,
            corner_radius=10,
            fg_color="#ec4899",
            hover_color="#db2777"
        )
        self.blur_btn.pack(fill="x", padx=20, pady=(0, 10))

        # 批次遮蔽按鈕
        self.batch_btn = ctk.CTkButton(
            right_frame,
            text="📦 批次遮蔽",
            command=self.batch_blur,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45,
            corner_radius=10,
            fg_color="#10b981",
            hover_color="#059669"
        )
        self.batch_btn.pack(fill="x", padx=20, pady=(0, 10))

        # 儲存結果按鈕
        self.save_btn = ctk.CTkButton(
            right_frame,
            text="💾 儲存結果",
            command=self.save_result,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45,
            corner_radius=10,
            fg_color="#8b5cf6",
            hover_color="#7c3aed"
        )
        self.save_btn.pack(fill="x", padx=20, pady=(0, 20))

        # ===== 底部狀態列 =====
        status_frame = ctk.CTkFrame(self.root, height=50, corner_radius=0)
        status_frame.pack(side="bottom", fill="x", padx=0, pady=0)
        status_frame.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="🟢 就緒 - 請上傳圖片",
            font=ctk.CTkFont(size=13),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=30, pady=10)

    def toggle_theme(self):
        """切換深色/淺色主題"""
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")

    def upload_image(self):
        """上傳圖片"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[
                ("圖片檔案", "*.jpg *.jpeg *.png *.bmp"),
                ("所有檔案", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.status_label.configure(text=f"✅ 已載入: {os.path.basename(file_path)}")

            # 清空之前的檢測結果
            self.current_faces = []
            self.face_info_text.delete("1.0", "end")
            self.face_info_text.insert("1.0", "請點選'檢測人臉'按鈕")

    def display_image(self, image_source, is_path=True):
        """顯示圖片（兼容性方法，用於初次上傳）

        Args:
            image_source: 圖片路徑或numpy陣列
            is_path: 是否為檔案路徑
        """
        try:
            if is_path:
                # 讀取圖片
                img = cv2.imread(image_source)
                if img is None:
                    raise ValueError("無法讀取圖片")
            else:
                # 已經是 numpy array
                img = image_source

            # 使用新的 Canvas 顯示方法
            self.display_image_on_canvas(img)

        except Exception as e:
            messagebox.showerror("錯誤", f"無法顯示圖片: {str(e)}")

    def detect_faces(self):
        """檢測人臉"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "請先上傳圖片")
            return

        try:
            self.status_label.configure(text="🔄 正在檢測人臉...")
            self.root.update()

            # 檢測人臉
            self.current_image, self.current_faces = self.blur_tool.detect_faces(
                self.current_image_path
            )

            if not self.current_faces:
                messagebox.showinfo("提示", "未檢測到人臉")
                self.status_label.configure(text="⚠️ 未檢測到人臉")
                return

            # 顯示檢測結果
            face_info = self.blur_tool.get_face_info(self.current_faces)
            self.face_info_text.delete("1.0", "end")
            self.face_info_text.insert("1.0", face_info)

            # 新增：預設全選所有人臉
            self.selected_face_ids = set(face["id"] for face in self.current_faces)

            # 使用新的互動式顯示
            self.update_selection_display()

            self.status_label.configure(
                text=f"✅ 檢測完成 - 發現 {len(self.current_faces)} 個人臉（已全選）"
            )

        except Exception as e:
            messagebox.showerror("錯誤", f"人臉檢測失敗: {str(e)}")
            self.status_label.configure(text="❌ 檢測失敗")


    def apply_blur(self):
        """執行遮蔽（基於選中的人臉）"""
        if not self.current_faces:
            messagebox.showwarning("警告", "請先檢測人臉")
            return

        if not self.selected_face_ids:
            messagebox.showwarning("警告", "請先選擇要遮蔽的人臉")
            return

        try:
            self.status_label.configure(text="🔄 正在遮蔽人臉...")
            self.root.update()

            # 使用選擇性遮蔽方法
            self.preview_image = self.blur_faces_selective(
                self.current_image,
                self.current_faces,
                self.selected_face_ids
            )

            # 顯示結果
            self.display_image_on_canvas(self.preview_image)

            self.status_label.configure(
                text=f"✅ 遮蔽完成 - 已遮蔽 {len(self.selected_face_ids)} 個人臉"
            )

        except Exception as e:
            messagebox.showerror("錯誤", f"遮蔽失敗: {str(e)}")

    def save_result(self):
        """儲存結果"""
        if self.preview_image is None:
            messagebox.showwarning("警告", "請先執行遮蔽")
            return

        # 選擇儲存位置
        file_path = filedialog.asksaveasfilename(
            title="儲存結果",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG圖片", "*.jpg"),
                ("PNG圖片", "*.png"),
                ("所有檔案", "*.*")
            ]
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.preview_image)
                messagebox.showinfo("成功", f"已儲存到: {file_path}")
                self.status_label.configure(text=f"💾 已儲存: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("錯誤", f"儲存失敗: {str(e)}")

    # ===== 工具選擇方法 =====
    def select_tool(self, tool):
        """選擇工具：pen 或 eraser"""
        self.current_tool = tool

        # 更新按鈕樣式
        if tool == "pen":
            self.pen_btn.configure(fg_color="#ec4899", hover_color="#db2777")
            self.eraser_btn.configure(fg_color="#6b7280", hover_color="#4b5563")
            self.status_label.configure(text="🖊️ 筆工具：點擊人臉以選擇遮蔽")
        else:
            self.pen_btn.configure(fg_color="#6b7280", hover_color="#4b5563")
            self.eraser_btn.configure(fg_color="#ec4899", hover_color="#db2777")
            self.status_label.configure(text="🧹 橡皮擦：點擊人臉以取消遮蔽")

    def select_all_faces(self):
        """全選所有人臉"""
        if not self.current_faces:
            messagebox.showwarning("警告", "請先檢測人臉")
            return

        self.selected_face_ids = set(face["id"] for face in self.current_faces)
        self.update_selection_display()
        self.status_label.configure(
            text=f"✅ 已全選 {len(self.selected_face_ids)} 個人臉"
        )

    def deselect_all_faces(self):
        """取消全選"""
        if not self.current_faces:
            messagebox.showwarning("警告", "請先檢測人臉")
            return

        self.selected_face_ids.clear()
        self.update_selection_display()
        self.status_label.configure(text="❌ 已取消所有選擇")

    # ===== 坐標轉換方法 =====
    def display_to_original_coords(self, display_x, display_y):
        """將顯示坐標轉換為原圖坐標"""
        if self.scale_x == 0 or self.scale_y == 0:
            return 0, 0

        original_x = int((display_x - self.offset_x) / self.scale_x)
        original_y = int((display_y - self.offset_y) / self.scale_y)

        return original_x, original_y

    def get_face_at_position(self, x, y):
        """獲取指定位置（原圖坐標）的人臉ID"""
        for face in self.current_faces:
            x1, y1, x2, y2 = face["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return face["id"]
        return None

    def view_selection(self):
        """查看當前選擇"""
        if not self.current_faces:
            messagebox.showwarning("警告", "請先檢測人臉")
            return

        self.update_selection_display()

        selected = len(self.selected_face_ids)
        total = len(self.current_faces)
        self.status_label.configure(
            text=f"👁️ 當前選擇: {selected}/{total} 個人臉"
        )

    # ===== Canvas 事件處理 =====
    def on_canvas_click(self, event):
        """Canvas 點擊事件"""
        if not self.current_faces:
            return

        # 轉換坐標
        original_x, original_y = self.display_to_original_coords(event.x, event.y)

        # 檢測點擊的人臉
        clicked_face_id = self.get_face_at_position(original_x, original_y)

        if clicked_face_id is not None:
            # 根據工具切換狀態
            if self.current_tool == "pen":
                self.selected_face_ids.add(clicked_face_id)
                action = "選中"
            else:  # eraser
                self.selected_face_ids.discard(clicked_face_id)
                action = "取消選擇"

            # 更新顯示
            self.update_selection_display()
            self.status_label.configure(
                text=f"{action}人臉 #{clicked_face_id} - 已選擇 {len(self.selected_face_ids)}/{len(self.current_faces)}"
            )

    def on_canvas_hover(self, event):
        """Canvas 懸停事件"""
        if not self.current_faces:
            return

        original_x, original_y = self.display_to_original_coords(event.x, event.y)
        hover_face_id = self.get_face_at_position(original_x, original_y)

        # 只有懸停的人臉改變時才更新
        if hover_face_id != self.hover_face_id:
            self.hover_face_id = hover_face_id
            self.update_selection_display()

    def on_canvas_leave(self, event):
        """Canvas 離開事件"""
        if self.hover_face_id is not None:
            self.hover_face_id = None
            self.update_selection_display()

    # ===== Canvas 顯示方法 =====
    def display_image_on_canvas(self, image_array):
        """在 Canvas 上顯示圖片（numpy array）"""
        # 獲取 Canvas 尺寸
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas 尚未初始化
            canvas_width = 900
            canvas_height = 680

        # 轉換為 PIL Image
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 計算縮放比例（保持長寬比）
        img_ratio = img_pil.width / img_pil.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            # 圖片更寬
            new_width = canvas_width
            new_height = int(new_width / img_ratio)
        else:
            # 圖片更高
            new_height = canvas_height
            new_width = int(new_height * img_ratio)

        # 儲存縮放參數
        self.scale_x = new_width / img_pil.width
        self.scale_y = new_height / img_pil.height

        # 計算偏移量（居中顯示）
        self.offset_x = (canvas_width - new_width) // 2
        self.offset_y = (canvas_height - new_height) // 2

        # 縮放圖片
        img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 轉換為 PhotoImage
        self.tk_image = ImageTk.PhotoImage(img_resized)

        # 清除舊圖片，顯示新圖片
        if self.canvas_image_id:
            self.canvas.delete(self.canvas_image_id)

        self.canvas_image_id = self.canvas.create_image(
            self.offset_x, self.offset_y,
            anchor=tk.NW,
            image=self.tk_image
        )

    # ===== 互動式人臉框繪製 =====
    def draw_interactive_boxes(self, img, faces, selected_ids, hover_id=None):
        """繪製互動式人臉框"""
        img_with_boxes = img.copy()

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_id = face["id"]

            # 決定顏色和粗細
            if face_id == hover_id:
                # 懸停：黃色粗框
                color = (0, 255, 255)  # BGR 黃色
                thickness = 4
            elif face_id in selected_ids:
                # 選中：紅色
                color = (0, 0, 255)  # BGR 紅色
                thickness = 3
            else:
                # 未選中：綠色
                color = (0, 255, 0)  # BGR 綠色
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

    def update_selection_display(self):
        """更新人臉框的顯示"""
        if not self.current_faces or self.current_image is None:
            return

        # 繪製人臉框
        img_with_boxes = self.draw_interactive_boxes(
            self.current_image,
            self.current_faces,
            self.selected_face_ids,
            self.hover_face_id
        )

        # 更新 Canvas 顯示
        self.display_image_on_canvas(img_with_boxes)

        # 更新選擇狀態標籤
        total = len(self.current_faces)
        selected = len(self.selected_face_ids)
        self.selection_status_label.configure(
            text=f"已選擇 {selected}/{total} 個人臉進行遮蔽"
        )

    # ===== 選擇性遮蔽方法 =====
    def blur_faces_selective(self, img, faces, selected_ids):
        """只遮蔽選中的人臉"""
        # 轉換為 PIL 圖片
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 載入字型
        font_size = 100
        try:
            font_paths = [
                "C:/Windows/Fonts/seguiemj.ttf",
                "C:/Windows/Fonts/NotoColorEmoji.ttf",
                "C:/Windows/Fonts/seguisym.ttf"
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # 只遮蔽選中的人臉
        emoji_index = 0
        emojis = ["😊", "🥰", "😄", "😃", "😁", "🤗", "😺", "😸"]

        for face in faces:
            face_id = face["id"]

            # 只處理選中的人臉
            if face_id in selected_ids:
                x1, y1, x2, y2 = face["bbox"]

                # 計算人臉中心點
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 計算 emoji 大小
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

                # 選擇 emoji
                emoji = emojis[emoji_index % len(emojis)]
                emoji_index += 1

                # 繪製 emoji
                bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2

                draw.text((text_x, text_y), emoji, font=emoji_font, embedded_color=True)

        # 轉換回 OpenCV 格式
        img_with_emoji = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return img_with_emoji

    # ===== 批次處理方法 =====
    def batch_blur(self):
        """批次遮蔽多張圖片"""
        # 1. 選擇多張圖片
        file_paths = filedialog.askopenfilenames(
            title="選擇要批次處理的圖片",
            filetypes=[
                ("圖片檔案", "*.jpg *.jpeg *.png *.bmp"),
                ("所有檔案", "*.*")
            ]
        )

        if not file_paths:
            return

        # 2. 警告使用者
        result = messagebox.askokcancel(
            "批次遮蔽確認",
            f"即將批次處理 {len(file_paths)} 張圖片\n\n"
            "⚠️ 警告：批次模式會自動遮蔽所有檢測到的人臉\n"
            "處理後的圖片將保存在原圖片目錄，檔名加上 _blurred 後綴\n\n"
            "確定要繼續嗎？"
        )

        if not result:
            return

        # 3. 創建進度視窗
        progress_window = self.create_progress_window(len(file_paths))

        # 4. 處理每張圖片
        success_count = 0
        error_files = []

        for idx, file_path in enumerate(file_paths, 1):
            try:
                # 更新進度
                self.update_progress(
                    progress_window, idx, len(file_paths),
                    os.path.basename(file_path)
                )

                # 檢測人臉
                img, faces = self.blur_tool.detect_faces(file_path)

                if not faces:
                    continue  # 跳過無人臉圖片

                # 遮蔽所有人臉
                blurred_img = self.blur_tool.blur_faces_with_emoji(
                    img, faces, 1, len(faces)
                )

                # 生成輸出路徑
                dir_name = os.path.dirname(file_path)
                base_name = os.path.basename(file_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(dir_name, f"{name}_blurred{ext}")

                # 儲存
                cv2.imwrite(output_path, blurred_img)
                success_count += 1

            except Exception as e:
                error_files.append((file_path, str(e)))

            self.root.update()  # 保持 UI 響應

        # 5. 關閉進度視窗並顯示結果
        progress_window.destroy()

        if error_files:
            error_msg = "\n".join([f"- {os.path.basename(f)}: {e}"
                                   for f, e in error_files[:5]])  # 只顯示前5個錯誤
            if len(error_files) > 5:
                error_msg += f"\n... 還有 {len(error_files) - 5} 個錯誤"

            messagebox.showwarning(
                "批次處理完成",
                f"成功處理: {success_count}/{len(file_paths)} 張圖片\n\n"
                f"失敗 {len(error_files)} 個:\n{error_msg}"
            )
        else:
            messagebox.showinfo(
                "批次處理完成",
                f"✅ 成功處理 {success_count}/{len(file_paths)} 張圖片"
            )

        self.status_label.configure(
            text=f"✅ 批次處理完成 - 成功 {success_count}/{len(file_paths)}"
        )

    def create_progress_window(self, total_files):
        """創建進度視窗"""
        # 創建頂層視窗
        progress_win = tk.Toplevel(self.root)
        progress_win.title("批次處理進度")
        progress_win.geometry("500x200")
        progress_win.resizable(False, False)

        # 置中
        progress_win.transient(self.root)
        progress_win.grab_set()

        # 標題
        tk.Label(
            progress_win,
            text="正在批次處理圖片...",
            font=("Arial", 14, "bold")
        ).pack(pady=(20, 10))

        # 當前檔案標籤
        progress_win.current_file_label = tk.Label(
            progress_win,
            text="準備中...",
            font=("Arial", 11),
            wraplength=450
        )
        progress_win.current_file_label.pack(pady=5)

        # 進度條
        progress_win.progress_bar = ttk.Progressbar(
            progress_win,
            length=450,
            mode='determinate',
            maximum=total_files
        )
        progress_win.progress_bar.pack(pady=10)

        # 進度文字
        progress_win.progress_label = tk.Label(
            progress_win,
            text=f"0/{total_files}",
            font=("Arial", 11)
        )
        progress_win.progress_label.pack(pady=5)

        return progress_win

    def update_progress(self, progress_window, current, total, filename):
        """更新進度視窗"""
        progress_window.current_file_label.configure(
            text=f"正在處理: {filename}"
        )
        progress_window.progress_bar['value'] = current
        progress_window.progress_label.configure(
            text=f"{current}/{total}"
        )
        progress_window.update()


def main():
    """主函式"""
    root = ctk.CTk()
    app = ModernFaceBlurGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
