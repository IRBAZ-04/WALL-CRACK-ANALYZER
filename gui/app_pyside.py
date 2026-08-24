"""
gui/app_pyside.py
------------------
State-of-the-Art Desktop GUI for Automatic Wall Crack Detection, Measurement,
Damage Assessment, and Algorithm Comparison (UCSC313L Project).

Built with PySide6 (Qt6) featuring a Cyber/Nordic dark glassmorphism theme,
interactive side-by-side comparison slider, live webcam feed integration,
in-app ground-truth mask annotation painter, real-time measurements HUD,
and benchmark analytics dashboard.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal, QSize, QPoint, QRectF
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen, QBrush, QPainterPath
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QStackedWidget, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QFrame, QSplitter, QScrollArea, QTabWidget, QDialog, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QHeaderView, QRadioButton, QButtonGroup
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Import project pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as pipeline
from evaluation import comparison


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QFrame.Sidebar {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

QFrame.Card {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
}

QLabel.HeaderTitle {
    font-size: 18px;
    font-weight: bold;
    color: #38bdf8;
}

QLabel.CardTitle {
    font-size: 14px;
    font-weight: bold;
    color: #f0f6fc;
}

QLabel.KpiValue {
    font-size: 22px;
    font-weight: bold;
    color: #38bdf8;
}

QLabel.KpiLabel {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
}

QPushButton.NavButton {
    background-color: transparent;
    color: #8b949e;
    border: none;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
}

QPushButton.NavButton:hover {
    background-color: #21262d;
    color: #f0f6fc;
}

QPushButton.NavButton:checked {
    background-color: #1f6beb;
    color: #ffffff;
}

QPushButton.PrimaryButton {
    background-color: #238636;
    color: #ffffff;
    border: 1px solid #2ea043;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton.PrimaryButton:hover {
    background-color: #2ea043;
}

QPushButton.SecondaryButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton.SecondaryButton:hover {
    background-color: #30363d;
    color: #f0f6fc;
}

QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 10px;
}

QTableWidget {
    background-color: #161b22;
    gridline-color: #21262d;
    border: 1px solid #21262d;
    border-radius: 6px;
}

QHeaderView::section {
    background-color: #21262d;
    color: #f0f6fc;
    padding: 6px;
    font-weight: bold;
    border: 1px solid #30363d;
}

QTabWidget::pane {
    border: 1px solid #21262d;
    background-color: #161b22;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #0d1117;
    color: #8b949e;
    padding: 8px 16px;
    border: 1px solid #21262d;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #161b22;
    color: #38bdf8;
    font-weight: bold;
}
"""


# ---------------- WEBCAM WORKER THREAD ----------------
class CameraThread(QThread):
    frame_captured = Signal(np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.frame_captured.emit(frame)
            self.msleep(30)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ---------------- WEBCAM CAPTURE DIALOG ----------------
class LiveCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Campus Wall Camera Capture")
        self.resize(700, 550)
        self.captured_frame = None

        layout = QVBoxLayout(self)

        self.video_label = QLabel("Initializing webcam feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 8px;")
        layout.addWidget(self.video_label, 1)

        btn_layout = QHBoxLayout()
        self.btn_snap = QPushButton("📷 Snap Photo", self)
        self.btn_snap.setProperty("class", "PrimaryButton")
        self.btn_snap.clicked.connect(self._snap)
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_snap)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.thread = CameraThread(0)
        self.thread.frame_captured.connect(self._update_frame)
        self.thread.start()

    def _update_frame(self, frame):
        self.current_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def _snap(self):
        if hasattr(self, "current_frame"):
            self.captured_frame = self.current_frame.copy()
            self.thread.stop()
            self.accept()

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)


# ---------------- INTERACTIVE MAGNIFYING LOUPE VIEWER WIDGET ----------------
class MagnifyingLoupeViewer(QLabel):
    def __init__(self, placeholder_text="Upload or capture a wall image to start", parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.placeholder_text = placeholder_text
        self.bgr_image = None
        self.scaled_pixmap = None
        self.loupe_enabled = True
        self.mouse_pos = QPoint(-1000, -1000)
        self.loupe_radius = 80
        self.zoom_factor = 2.5
        self.is_hovering = False

        self.setAlignment(Qt.AlignCenter)
        self.setText(placeholder_text)
        self.setStyleSheet("color: #8b949e; background-color: #0d1117; border-radius: 6px;")

    def set_bgr_image(self, bgr_img):
        self.bgr_image = bgr_img
        self.setText("")
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self.bgr_image is None:
            return
        h, w, ch = self.bgr_image.shape
        rgb = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.scaled_pixmap = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.scaled_pixmap)

    def enterEvent(self, event):
        self.is_hovering = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.is_hovering = False
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.position().toPoint()
        self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.loupe_enabled or not self.is_hovering or self.bgr_image is None or self.scaled_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pw, ph = self.scaled_pixmap.width(), self.scaled_pixmap.height()
        lw, lh = self.width(), self.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2

        mx = self.mouse_pos.x()
        my = self.mouse_pos.y()

        if not (ox <= mx <= ox + pw and oy <= my <= oy + ph):
            return

        oh, ow = self.bgr_image.shape[:2]
        img_x = int((mx - ox) * (ow / max(pw, 1)))
        img_y = int((my - oy) * (oh / max(ph, 1)))

        rw = int(self.loupe_radius * 2 / self.zoom_factor * (ow / max(pw, 1)))
        rh = int(self.loupe_radius * 2 / self.zoom_factor * (oh / max(ph, 1)))

        x1 = max(0, img_x - rw // 2)
        y1 = max(0, img_y - rh // 2)
        x2 = min(ow, x1 + rw)
        y2 = min(oh, y1 + rh)

        roi = self.bgr_image[y1:y2, x1:x2]
        if roi.size == 0:
            return

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rh_r, rw_r, _ = roi_rgb.shape
        q_roi = QImage(roi_rgb.data, rw_r, rh_r, rw_r * 3, QImage.Format_RGB888)
        pix_roi = QPixmap.fromImage(q_roi).scaled(
            self.loupe_radius * 2, self.loupe_radius * 2, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        path = QPainterPath()
        path.addEllipse(QRectF(
            mx - self.loupe_radius,
            my - self.loupe_radius,
            self.loupe_radius * 2,
            self.loupe_radius * 2
        ))

        painter.save()
        painter.setClipPath(path)
        painter.drawPixmap(
            int(mx - self.loupe_radius),
            int(my - self.loupe_radius),
            pix_roi
        )

        pen_reticle = QPen(QColor(56, 189, 248, 200), 1, Qt.DashLine)
        painter.setPen(pen_reticle)
        painter.drawLine(mx - 15, my, mx + 15, my)
        painter.drawLine(mx, my - 15, mx, my + 15)
        painter.restore()

        pen_ring = QPen(QColor(56, 189, 248), 3)
        painter.setPen(pen_ring)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(
            mx - self.loupe_radius,
            my - self.loupe_radius,
            self.loupe_radius * 2,
            self.loupe_radius * 2
        ))

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setBrush(QBrush(QColor(13, 17, 23, 220)))
        badge = QRectF(mx - 45, my + self.loupe_radius + 6, 90, 20)
        painter.drawRoundedRect(badge, 4, 4)
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.drawText(badge, Qt.AlignCenter, f"{self.zoom_factor:.1f}x LOUPE")


# ---------------- INTERACTIVE GROUND TRUTH ANNOTATION WIDGET ----------------
class GroundTruthPainter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_pixmap = None
        self.mask_image = None
        self.brush_size = 8
        self.drawing = False
        self.mode = "draw"  # 'draw' or 'erase'
        self.last_point = QPoint()

    def set_base_image(self, bgr_image):
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.bg_pixmap = QPixmap.fromImage(qimg)

        self.mask_image = QImage(w, h, QImage.Format_ARGB32)
        self.mask_image.fill(Qt.transparent)
        self.setFixedSize(w, h)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.mask_image:
            self.drawing = True
            self.last_point = event.position().toPoint()
            self._paint_to(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing and self.mask_image:
            self._paint_to(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def _paint_to(self, point):
        painter = QPainter(self.mask_image)
        if self.mode == "draw":
            pen = QPen(QColor(255, 0, 0, 200), self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        else:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(Qt.transparent, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.last_point, point)
        self.last_point = point
        painter.end()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.bg_pixmap:
            painter.drawPixmap(0, 0, self.bg_pixmap)
        if self.mask_image:
            painter.drawImage(0, 0, self.mask_image)

    def get_binary_mask(self):
        if not self.mask_image:
            return None
        w, h = self.mask_image.width(), self.mask_image.height()
        ptr = self.mask_image.bits()
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        # Mask where red alpha > 0
        binary = np.zeros((h, w), dtype=np.uint8)
        binary[arr[:, :, 3] > 50] = 255
        return binary


# ---------------- SIDE-BY-SIDE SPLIT COMPARISON WIDGET ----------------
class SplitComparisonViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_left = None
        self.img_right = None
        self.split_pos = 0.5
        self.dragging = False
        self.setMouseTracking(True)

    def set_images(self, left_bgr, right_bgr):
        h, w = left_bgr.shape[:2]
        q_left = QImage(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).data, w, h, w * 3, QImage.Format_RGB888)
        q_right = QImage(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB).data, w, h, w * 3, QImage.Format_RGB888)
        self.pix_left = QPixmap.fromImage(q_left)
        self.pix_right = QPixmap.fromImage(q_right)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self._update_split(event.position().x())

    def mouseMoveEvent(self, event):
        if self.dragging:
            self._update_split(event.position().x())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def _update_split(self, x):
        self.split_pos = max(0.0, min(1.0, x / max(self.width(), 1)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        if not hasattr(self, "pix_left") or not self.pix_left:
            painter.drawText(self.rect(), Qt.AlignCenter, "No images loaded for split comparison")
            return

        scaled_left = self.pix_left.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_right = self.pix_right.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        split_x = int(w * self.split_pos)

        # Draw left image portion
        painter.save()
        painter.setClipRect(0, 0, split_x, h)
        painter.drawPixmap(0, 0, scaled_left)
        painter.restore()

        # Draw right image portion
        painter.save()
        painter.setClipRect(split_x, 0, w - split_x, h)
        painter.drawPixmap(0, 0, scaled_right)
        painter.restore()

        # Draw handle line
        pen = QPen(QColor(56, 189, 248), 3)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, h)

        # Draw handle pill
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawEllipse(QPoint(split_x, h // 2), 12, 12)


# ---------------- MAIN APPLICATION WINDOW ----------------
class WallCrackMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUTOMATIC WALL CRACK DETECTOR & ANALYZER — UCSC313L")
        self.resize(1280, 800)

        self.image_path = None
        self.image_bgr = None
        self.image_id = None
        self.measurements = None
        self.current_algo = "canny"
        self.intermediate_stage_images = {}

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        sidebar = QFrame()
        sidebar.setProperty("class", "Sidebar")
        sidebar.setFixedWidth(230)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(12, 20, 12, 20)

        logo = QLabel("CRACK ANALYZER")
        logo.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8; letter-spacing: 1px;")
        subtitle = QLabel("Classical OpenCV Engine")
        subtitle.setStyleSheet("font-size: 10px; color: #8b949e; margin-bottom: 20px;")
        sb_layout.addWidget(logo)
        sb_layout.addWidget(subtitle)

        self.nav_group = QButtonGroup(self)
        self.btn_nav_home = QPushButton("🏠 Home & Controls")
        self.btn_nav_stages = QPushButton("🔬 Processing Stages")
        self.btn_nav_measure = QPushButton("📊 Measurements HUD")
        self.btn_nav_compare = QPushButton("📈 Algorithm Benchmark")
        self.btn_nav_3d = QPushButton("🏔️ 3D Surface Relief")
        self.btn_nav_annotate = QPushButton("✏️ Ground Truth Tool")
        self.btn_nav_demo = QPushButton("🎥 Live Demo Guide")

        nav_buttons = [
            (self.btn_nav_home, 0),
            (self.btn_nav_stages, 1),
            (self.btn_nav_measure, 2),
            (self.btn_nav_3d, 3),
            (self.btn_nav_compare, 4),
            (self.btn_nav_annotate, 5),
            (self.btn_nav_demo, 6),
        ]

        for btn, idx in nav_buttons:
            btn.setCheckable(True)
            btn.setProperty("class", "NavButton")
            self.nav_group.addButton(btn, idx)
            sb_layout.addWidget(btn)

        self.btn_nav_home.setChecked(True)
        self.nav_group.idClicked.connect(self._change_page)

        sb_layout.addStretch(1)

        status_card = QFrame()
        status_card.setProperty("class", "Card")
        sc_layout = QVBoxLayout(status_card)
        sc_layout.setContentsMargins(8, 8, 8, 8)
        self.lbl_sb_status = QLabel("Ready")
        self.lbl_sb_status.setStyleSheet("color: #22c55e; font-weight: bold; font-size: 11px;")
        sc_layout.addWidget(QLabel("STATUS", styleSheet="font-size: 9px; color: #8b949e;"))
        sc_layout.addWidget(self.lbl_sb_status)
        sb_layout.addWidget(status_card)

        main_layout.addWidget(sidebar)

        # --- STACKED PAGES ---
        self.pages_stack = QStackedWidget()
        main_layout.addWidget(self.pages_stack, 1)

        self._build_home_page()
        self._build_stages_page()
        self._build_measure_page()
        self._build_3d_page()
        self._build_compare_page()
        self._build_annotate_page()
        self._build_demo_page()

    def _change_page(self, page_idx):
        self.pages_stack.setCurrentIndex(page_idx)

    # ---------------- PAGE 0: HOME & PIPELINE CONTROLS ----------------
    def _build_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Image Input & Edge Detection Control")
        header.setProperty("class", "HeaderTitle")
        layout.addWidget(header)

        top_bar = QHBoxLayout()

        self.btn_upload = QPushButton("📂 Upload Wall Photo")
        self.btn_upload.setProperty("class", "PrimaryButton")
        self.btn_upload.clicked.connect(self._upload_photo)

        self.btn_camera = QPushButton("🎥 Live Campus Camera")
        self.btn_camera.setProperty("class", "SecondaryButton")
        self.btn_camera.clicked.connect(self._open_webcam)

        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["canny", "sobel", "prewitt", "laplacian", "fusion"])
        self.combo_algo.currentTextChanged.connect(self._set_algo)

        self.btn_analyze = QPushButton("⚡ Run Complete Pipeline")
        self.btn_analyze.setProperty("class", "PrimaryButton")
        self.btn_analyze.setStyleSheet("background-color: #1f6beb;")
        self.btn_analyze.clicked.connect(self._run_pipeline)

        top_bar.addWidget(self.btn_upload)
        top_bar.addWidget(self.btn_camera)
        top_bar.addWidget(QLabel("Edge Operator:"))
        top_bar.addWidget(self.combo_algo)
        top_bar.addWidget(self.btn_analyze)
        top_bar.addStretch(1)

        layout.addLayout(top_bar)

        # Main Split View (Preview & Realtime Parameters)
        content_split = QHBoxLayout()

        # Preview Area
        preview_card = QFrame()
        preview_card.setProperty("class", "Card")
        pv_layout = QVBoxLayout(preview_card)
        self.lbl_home_preview = MagnifyingLoupeViewer("Upload or capture a campus wall image to start")
        pv_layout.addWidget(self.lbl_home_preview)

        content_split.addWidget(preview_card, 2)

        # Quick Control Panel
        param_card = QFrame()
        param_card.setProperty("class", "Card")
        param_card.setFixedWidth(300)
        pm_layout = QVBoxLayout(param_card)
        lbl_tuning = QLabel("Pipeline Tuning & Calibration")
        lbl_tuning.setProperty("class", "CardTitle")
        pm_layout.addWidget(lbl_tuning)

        pm_layout.addWidget(QLabel("Calibration Reference (cm):"))
        self.spin_calib_cm = QDoubleSpinBox()
        self.spin_calib_cm.setValue(10.0)
        self.spin_calib_cm.setRange(1.0, 100.0)
        pm_layout.addWidget(self.spin_calib_cm)

        pm_layout.addWidget(QLabel("CLAHE Clip Limit:"))
        self.slider_clahe = QSlider(Qt.Horizontal)
        self.slider_clahe.setRange(10, 50)
        self.slider_clahe.setValue(20)
        pm_layout.addWidget(self.slider_clahe)

        pm_layout.addWidget(QLabel("Canny Upper Thresh Ratio:"))
        self.slider_canny = QSlider(Qt.Horizontal)
        self.slider_canny.setRange(50, 150)
        self.slider_canny.setValue(100)
        pm_layout.addWidget(self.slider_canny)

        pm_layout.addStretch(1)

        content_split.addWidget(param_card, 1)
        layout.addLayout(content_split, 1)

        self.pages_stack.addWidget(page)

    # ---------------- PAGE 1: PROCESSING STAGES & SPLIT COMPARISON ----------------
    def _build_stages_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Intermediate Processing Stages & Split Viewer")
        header.setProperty("class", "HeaderTitle")
        layout.addWidget(header)

        self.tab_stages = QTabWidget()

        # Tab 1: Split Slider
        tab_split = QWidget()
        split_lay = QVBoxLayout(tab_split)
        self.split_viewer = SplitComparisonViewer()
        split_lay.addWidget(self.split_viewer)
        self.tab_stages.addTab(tab_split, "2-Up Split Comparison Slider")

        # Tab 2: All 14 Intermediate Stages Grid
        tab_grid = QWidget()
        grid_lay = QVBoxLayout(tab_grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_container = QWidget()
        self.grid_layout = QHBoxLayout(grid_container)
        scroll.setWidget(grid_container)
        grid_lay.addWidget(scroll)
        self.tab_stages.addTab(tab_grid, "14-Stage Pipeline Grid")

        layout.addWidget(self.tab_stages)
        self.pages_stack.addWidget(page)

    # ---------------- PAGE 2: MEASUREMENTS HUD ----------------
    def _build_measure_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Crack Quantification & Damage Assessment HUD")
        header.setProperty("class", "HeaderTitle")
        layout.addWidget(header)

        # KPI Dashboard Cards Row
        kpi_row = QHBoxLayout()

        self.kpi_crack_status = self._create_kpi_card("CRACK DETECTED", "NO", "#ef4444")
        self.kpi_confidence = self._create_kpi_card("CONFIDENCE", "--", "#38bdf8")
        self.kpi_length = self._create_kpi_card("ESTIMATED LENGTH", "--", "#22c55e")
        self.kpi_width = self._create_kpi_card("AVG WIDTH", "--", "#38bdf8")
        self.kpi_damage = self._create_kpi_card("DAMAGE WALL %", "--", "#f59e0b")

        kpi_row.addWidget(self.kpi_crack_status)
        kpi_row.addWidget(self.kpi_confidence)
        kpi_row.addWidget(self.kpi_length)
        kpi_row.addWidget(self.kpi_width)
        kpi_row.addWidget(self.kpi_damage)

        layout.addLayout(kpi_row)

        # Annotated Image & Detail Breakdown
        hud_body = QHBoxLayout()

        self.lbl_hud_annotated = MagnifyingLoupeViewer("Run Pipeline to view annotated overlay")
        hud_body.addWidget(self.lbl_hud_annotated, 2)

        detail_card = QFrame()
        detail_card.setProperty("class", "Card")
        dc_layout = QVBoxLayout(detail_card)
        lbl_breakdown = QLabel("Full Metric Breakdown")
        lbl_breakdown.setProperty("class", "CardTitle")
        dc_layout.addWidget(lbl_breakdown)

        self.txt_hud_details = QLabel("No measurements computed yet.")
        self.txt_hud_details.setWordWrap(True)
        self.txt_hud_details.setStyleSheet("font-family: Consolas, monospace; color: #c9d1d9; font-size: 12px;")
        dc_layout.addWidget(self.txt_hud_details)
        dc_layout.addStretch(1)

        hud_body.addWidget(detail_card, 1)
        layout.addLayout(hud_body, 1)

        self.pages_stack.addWidget(page)

    def _create_kpi_card(self, title, default_val, color="#38bdf8"):
        card = QFrame()
        card.setProperty("class", "Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lbl_t = QLabel(title)
        lbl_t.setProperty("class", "KpiLabel")
        lbl_v = QLabel(default_val)
        lbl_v.setProperty("class", "KpiValue")
        lbl_v.setStyleSheet(f"color: {color};")
        lay.addWidget(lbl_t)
        lay.addWidget(lbl_v)
        card.value_label = lbl_v
        return card

    # ---------------- PAGE 3: 3D SURFACE RELIEF MESH ----------------
    def _build_3d_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("3D Structural Elevation & Fracture Relief Landscape")
        header.setProperty("class", "HeaderTitle")
        layout.addWidget(header)

        control_bar = QHBoxLayout()
        self.btn_render_3d = QPushButton("🏔️ Generate 3D Surface Relief")
        self.btn_render_3d.setProperty("class", "PrimaryButton")
        self.btn_render_3d.clicked.connect(self._plot_3d_mesh)

        self.combo_3d_style = QComboBox()
        self.combo_3d_style.addItems(["Surface Mesh", "Wireframe", "Depth Contour Lines"])
        self.combo_3d_style.currentTextChanged.connect(lambda _: self._plot_3d_mesh())

        control_bar.addWidget(self.btn_render_3d)
        control_bar.addWidget(QLabel("3D Style:"))
        control_bar.addWidget(self.combo_3d_style)
        control_bar.addStretch(1)

        layout.addLayout(control_bar)

        # 3D Matplotlib Canvas
        from mpl_toolkits.mplot3d import Axes3D
        self.fig_3d = plt.figure(figsize=(8, 6))
        self.fig_3d.patch.set_facecolor('#161b22')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        layout.addWidget(self.canvas_3d, 1)

        self.pages_stack.addWidget(page)

    def _plot_3d_mesh(self):
        if self.image_bgr is None:
            QMessageBox.warning(self, "No Image", "Upload or capture an image first!")
            return

        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')
        ax.set_facecolor('#0d1117')

        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        step = max(1, min(h, w) // 90)

        sub_gray = gray[::step, ::step]
        sh, sw = sub_gray.shape

        X = np.arange(0, sw, 1)
        Y = np.arange(0, sh, 1)
        X, Y = np.meshgrid(X, Y)

        Z = sub_gray.astype(np.float32)

        style = self.combo_3d_style.currentText()
        if style == "Surface Mesh":
            surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.92)
            self.fig_3d.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        elif style == "Wireframe":
            ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='#38bdf8', linewidth=0.6)
        else:
            ax.contour3D(X, Y, Z, 25, cmap='binary')

        ax.set_title("3D Crack Depth Elevation Relief Landscape", color='#f0f6fc', fontsize=12)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        self.canvas_3d.draw()

    # ---------------- PAGE 3: ALGORITHM BENCHMARK ----------------
    def _build_compare_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Algorithm Evaluation Benchmark (Sobel vs Prewitt vs Laplacian vs Canny vs Fusion)")
        header.setProperty("class", "HeaderTitle")
        layout.addWidget(header)

        btn_run_eval = QPushButton("🔄 Re-run Batch Evaluation on Dataset")
        btn_run_eval.setProperty("class", "PrimaryButton")
        btn_run_eval.clicked.connect(self._reload_evaluation)
        layout.addWidget(btn_run_eval)

        # Table & Chart Split
        split = QHBoxLayout()

        self.tbl_benchmark = QTableWidget()
        self.tbl_benchmark.setColumnCount(6)
        self.tbl_benchmark.setHorizontalHeaderLabels([
            "Algorithm", "Precision", "Recall", "F1 Score", "Accuracy", "Time (sec)"
        ])
        self.tbl_benchmark.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        split.addWidget(self.tbl_benchmark, 1)

        # Matplotlib Chart Canvas
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.figure.patch.set_facecolor('#161b22')
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#c9d1d9')
        self.ax.spines['bottom'].set_color('#21262d')
        self.ax.spines['left'].set_color('#21262d')
        self.canvas = FigureCanvas(self.figure)

        split.addWidget(self.canvas, 1)

        layout.addLayout(split, 1)
        self.pages_stack.addWidget(page)

    # ---------------- PAGE 4: GROUND TRUTH ANNOTATION TOOL ----------------
    def _build_annotate_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("In-App Ground-Truth Crack Mask Annotation Tool")
        header.setProperty("class", "HeaderTitle")
        layout.addWidget(header)

        tool_bar = QHBoxLayout()
        btn_draw = QRadioButton("🖌️ Draw Crack (White)")
        btn_draw.setChecked(True)
        btn_erase = QRadioButton("🧹 Erase (Background)")

        btn_draw.toggled.connect(lambda: setattr(self.painter_widget, "mode", "draw" if btn_draw.isChecked() else "erase"))

        btn_save_gt = QPushButton("💾 Save Ground Truth Mask")
        btn_save_gt.setProperty("class", "PrimaryButton")
        btn_save_gt.clicked.connect(self._save_ground_truth_mask)

        tool_bar.addWidget(btn_draw)
        tool_bar.addWidget(btn_erase)
        tool_bar.addWidget(btn_save_gt)
        tool_bar.addStretch(1)

        layout.addLayout(tool_bar)

        self.painter_widget = GroundTruthPainter()
        scroll = QScrollArea()
        scroll.setWidget(self.painter_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        self.pages_stack.addWidget(page)

    # ---------------- PAGE 5: LIVE DEMO PRESENTATION GUIDE ----------------
    def _build_demo_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("10-Minute Live Campus Presentation Guide (Faculty Defense)")
        header.setProperty("class", "HeaderTitle")
        layout.addWidget(header)

        demo_text = QLabel(
            "<b>Minute 0-1:</b> Problem statement, motivation, structural safety importance.<br>"
            "<b>Minute 1-2:</b> Capture live wall image on campus via webcam / file upload.<br>"
            "<b>Minute 2-4:</b> Step through preprocessing & Compare Sobel, Prewitt, Laplacian, Canny.<br>"
            "<b>Minute 4-6:</b> Demonstrate Otsu/Adaptive binarization, morphological gap filling & skeletonization.<br>"
            "<b>Minute 6-8:</b> Review Measurements HUD: Length, Min/Max Width (Distance Transform), Wall Damage %.<br>"
            "<b>Minute 8-9:</b> Present Algorithm Benchmark table & F1-score comparison graph.<br>"
            "<b>Minute 9-10:</b> Explain why Fusion/Canny performed best, state physical calibration limitations & Q&A."
        )
        demo_text.setWordWrap(True)
        demo_text.setStyleSheet("font-size: 14px; color: #c9d1d9; background-color: #161b22; padding: 20px; border-radius: 8px;")
        layout.addWidget(demo_text)
        layout.addStretch(1)

        self.pages_stack.addWidget(page)

    # ---------------- ACTION HANDLERS ----------------
    def _upload_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Campus Wall Photo", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        self.image_path = path
        self.image_id = os.path.splitext(os.path.basename(path))[0]
        self.image_bgr = cv2.imread(path)

        # Show preview
        self._display_image_on_label(self.image_bgr, self.lbl_home_preview)
        self.lbl_sb_status.setText(f"Loaded: {os.path.basename(path)}")
        self.painter_widget.set_base_image(self.image_bgr)

    def _open_webcam(self):
        dialog = LiveCameraDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.captured_frame is not None:
            self.image_bgr = dialog.captured_frame
            self.image_id = "CAM_LIVE_CAPTURE"
            self.image_path = os.path.join("dataset", "raw", "CAM_LIVE_CAPTURE.jpg")
            os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
            cv2.imwrite(self.image_path, self.image_bgr)

            self._display_image_on_label(self.image_bgr, self.lbl_home_preview)
            self.lbl_sb_status.setText("Live webcam frame captured!")
            self.painter_widget.set_base_image(self.image_bgr)

    def _set_algo(self, algo_name):
        self.current_algo = algo_name

    def _run_pipeline(self):
        if self.image_bgr is None and not self.image_path:
            QMessageBox.warning(self, "No Image", "Please upload or capture a wall image first!")
            return

        self.lbl_sb_status.setText("Running processing pipeline...")
        QApplication.processEvents()

        try:
            ref_cm = self.spin_calib_cm.value()
            final_img, measurements = pipeline.run_pipeline(
                self.image_path, chosen_algo=self.current_algo, reference_cm=ref_cm
            )
            self.measurements = measurements
            self.lbl_sb_status.setText("Pipeline completed successfully!")

            # Update Measurements HUD
            self._update_measurements_hud(final_img, measurements)

            # Update Split Viewer
            if os.path.exists(self.image_path):
                raw = cv2.imread(self.image_path)
                self.split_viewer.set_images(raw, final_img)

            # Reload evaluation metrics table
            self._reload_evaluation()

            QMessageBox.information(self, "Analysis Done", "Pipeline processing finished! Check Measurements HUD and Processing Stages.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Pipeline failure: {str(e)}")
            self.lbl_sb_status.setText("Pipeline error!")

    def _update_measurements_hud(self, final_img, m):
        self._display_image_on_label(final_img, self.lbl_hud_annotated)

        crack_found = m["area"]["crack_area_px"] > 0
        self.kpi_crack_status.value_label.setText("YES" if crack_found else "NO")
        self.kpi_crack_status.value_label.setStyleSheet("color: #22c55e;" if crack_found else "color: #ef4444;")

        self.kpi_confidence.value_label.setText(f"{m['confidence']}%")
        self.kpi_length.value_label.setText(
            f"{m['length']['length_cm']} cm" if m['length']['length_cm'] else f"{m['length']['length_px']} px"
        )
        self.kpi_width.value_label.setText(
            f"{m['width']['avg_width_cm']} cm" if m['width']['avg_width_cm'] else f"{m['width']['avg_width_px']} px"
        )
        self.kpi_damage.value_label.setText(f"{m['area']['damage_percentage']}%")

        w_m = m["width"]
        details_text = (
            f"Algorithm Selected: {self.current_algo.upper()}\n"
            f"Lighting Detected : {m['lighting_detected']}\n"
            f"Segmentation Used : {m['segmentation_method']}\n"
            f"Calibrated (Marker): {'YES' if m['calibrated'] else 'NO (px only)'}\n\n"
            f"STRUCTURAL RISK   : {w_m.get('severity_level', 'N/A')}\n"
            f"ORIENTATION       : {w_m.get('orientation_type', 'N/A')} ({w_m.get('orientation_deg', 0.0)}°)\n\n"
            f"Crack Area (px)   : {m['area']['crack_area_px']} px\n"
            f"Wall Area (px)    : {m['area']['wall_area_px']} px\n"
            f"Damage Percentage : {m['area']['damage_percentage']}%"
        )
        self.txt_hud_details.setText(details_text)

    def _reload_evaluation(self):
        summary = comparison.run_evaluation()
        if summary is None:
            return

        self.tbl_benchmark.setRowCount(len(summary))
        for r_idx, row in summary.iterrows():
            self.tbl_benchmark.setItem(r_idx, 0, QTableWidgetItem(str(row["algorithm"])))
            self.tbl_benchmark.setItem(r_idx, 1, QTableWidgetItem(str(row["precision"])))
            self.tbl_benchmark.setItem(r_idx, 2, QTableWidgetItem(str(row["recall"])))
            self.tbl_benchmark.setItem(r_idx, 3, QTableWidgetItem(str(row["f1_score"])))
            self.tbl_benchmark.setItem(r_idx, 4, QTableWidgetItem(str(row["accuracy"])))
            self.tbl_benchmark.setItem(r_idx, 5, QTableWidgetItem(str(row["processing_time_sec"])))

        # Plot Matplotlib Chart
        self.ax.clear()
        algos = summary["algorithm"]
        f1s = summary["f1_score"]
        bars = self.ax.bar(algos, f1s, color="#38bdf8")
        self.ax.set_title("Algorithm F1-Score Comparison", color="#f0f6fc", fontsize=12)
        self.ax.set_ylabel("F1 Score", color="#c9d1d9")
        self.canvas.draw()

    def _save_ground_truth_mask(self):
        mask = self.painter_widget.get_binary_mask()
        if mask is None or not self.image_id:
            QMessageBox.warning(self, "No Mask", "Please paint a ground truth mask first!")
            return

        gt_dir = "dataset/ground_truth"
        os.makedirs(gt_dir, exist_ok=True)
        save_path = os.path.join(gt_dir, f"{self.image_id}.png")
        cv2.imwrite(save_path, mask)
        QMessageBox.information(self, "Saved", f"Ground truth mask saved to {save_path}")

    def _display_image_on_label(self, bgr_image, label):
        if isinstance(label, MagnifyingLoupeViewer):
            label.set_bgr_image(bgr_image)
        else:
            h, w, ch = bgr_image.shape
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    win = WallCrackMainWindow()
    win.show()
    sys.exit(app.exec())
