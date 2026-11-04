python launcher.pypython launcher.py# Automatic Wall Crack Detection, Measurement and Damage Assessment
### UCSC313L Image Processing Project (BCA Semester 5)

> **Core Highlight: 100% From-Scratch Mathematical Algorithms — Zero OpenCV Algorithm Dependencies!**  
> Every filter, convolution, gradient operator, edge detector, thresholding method, and morphological operator is implemented directly using pure mathematical matrix operations in **NumPy** and **Pillow (PIL)**.

---

## 🌟 Key Features & Extraordinary Novelty

1. **Pure Mathematical Image Processing from First Principles (Zero `cv2` in Algorithms)**:
   * **Vectorized 2D Discrete Convolution Engine** (`algorithms/convolution.py`): Stride-trick sliding window convolution with zero-padding and kernel flipping ($180^\circ$).
   * **Discrete Gradient Operators**: Discrete $K_x, K_y$ convolutions for **Sobel** ($\sqrt{G_x^2 + G_y^2}$) and **Prewitt** kernels.
   * **Laplacian of Gaussian (LoG)**: 2nd spatial derivative $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$ with 8-neighbor discrete stencil.
   * **Textbook 5-Stage Canny Edge Detector from Scratch** (`algorithms/canny.py`):
     1. Gaussian 2D smoothing filter
     2. Spatial gradient magnitude and direction $\theta = \arctan2(G_y, G_x)$
     3. Non-Maximum Suppression (NMS) with 4-sector angle quantization ($0^\circ, 45^\circ, 90^\circ, 135^\circ$)
     4. Double thresholding ($T_{high}, T_{low}$) using Otsu's optimal variance
     5. 8-connected Breadth-First Search (BFS) edge tracking by hysteresis
   * **From-Scratch Otsu's Global Binarization** (`algorithms/segmentation.py`): Maximizing between-class variance $\sigma_B^2(t) = \omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2$.
   * **Local Adaptive Window Thresholding**: Moving-window local mean calculation with Gaussian-weighted offset.
   * **Mathematical Morphology** (`algorithms/morphology.py`): Pure NumPy Dilation (moving window $\max$), Erosion (moving window $\min$), Morphological Closing, and 8-connected component flood-fill labeling.
   * **Eigenvector PCA Orientation** (`measurement/width.py`): Centered coordinate covariance matrix eigenvectors ($\mathbf{C} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$) to determine exact structural crack angle $\theta$.

2. **MobileNetV2 Deep Learning Intelligent Pre-Screening**:
   * Pre-screens incoming photos in real time.
   * If a wall is **NORMAL / SOUND**, the system immediately reports **"No Crack Detected"** and skips unnecessary computation.
   * If a **CRACK** is detected, it proceeds to the full classical measurement and severity assessment pipeline.

3. **Interactive Magnifying Glass Loupe Lens (Hover Inspection Tool)**:
   * Real-time $2.5\times$ magnified circular loupe lens with reticle crosshairs following the cursor on the PySide6 UI.

4. **🏔️ 3D Structural Elevation & Fracture Relief Landscape**:
   * Renders a 3D surface mesh plot (using Matplotlib 3D) where sound wall surfaces form high plateaus while crack fractures carve out deep structural ravines.

5. **2-Up Side-by-Side Split Image Comparison Slider**:
   * Interactive drag handle comparing original campus wall photos against processed crack edge overlays in real time.

6. **In-App Ground Truth Annotation Painter**:
   * Draw ground-truth crack masks directly on campus photos inside the desktop app using a brush tool to populate evaluation ground truth data.

7. **Structural Risk Rating Triage System**:
   * Evaluates crack width and damage percentage to assign color-coded severity levels: **LOW**, **MEDIUM**, **HIGH**, and **CRITICAL**.

---

## 🚀 How to Run the Project

### 1. Launch the Modern PySide6 Desktop GUI (Recommended for Presentation)
```powershell
.\.venv\Scripts\python.exe gui/app_pyside.py
```

### 2. Launch the Classic Tkinter GUI
```powershell
.\.venv\Scripts\python.exe gui/app.py
```

### 3. Run Single Image via Command-Line Interface (CLI)
```powershell
# Analyze a cracked wall with Canny (Pure Math from Scratch)
.\.venv\Scripts\python.exe main.py dataset/raw/CAMPUS_ACADEMIC_001.jpg canny

# Analyze with Multi-Operator Fusion
.\.venv\Scripts\python.exe main.py dataset/raw/CAMPUS_ACADEMIC_001.jpg fusion

# Screen a sound wall (skips pipeline with "No Crack Detected")
.\.venv\Scripts\python.exe main.py dataset/raw/CAMPUS_CLEAN_007.jpg canny
```

### 4. Run Automated AI Integration Test Suite
```powershell
.\.venv\Scripts\python.exe evaluation/test_ai_integration.py
```

### 5. Run Ground-Truth Benchmark & Generate Evaluation Charts
```powershell
.\.venv\Scripts\python.exe evaluation/comparison.py
```

---

## 📁 Project Architecture

```
WALL CRACK ANALYZER/
│
├── ai/                              # Deep Learning Screening Layer
│   ├── __init__.py
│   └── crack_classifier.py          # MobileNetV2 pre-screening engine (Pillow + TensorFlow)
│
├── algorithms/                      # 100% FROM-SCRATCH MATHEMATICAL ALGORITHMS (Zero OpenCV)
│   ├── __init__.py
│   ├── convolution.py               # Vectorized 2D spatial convolution & Gaussian kernel generator
│   ├── sobel.py                     # Discrete Sobel gradient operator (Gx, Gy, magnitude)
│   ├── prewitt.py                   # Discrete Prewitt gradient operator
│   ├── laplacian.py                 # Laplacian of Gaussian (LoG) 2nd spatial derivative
│   ├── canny.py                     # Complete 5-stage textbook Canny edge detector (NMS + Hysteresis)
│   ├── segmentation.py              # From-scratch Otsu's variance maximization & adaptive thresholding
│   ├── morphology.py                # Pure NumPy Dilation, Erosion, Closing & 8-connectivity labeling
│   ├── preprocessing.py             # Grayscale (BT.601), CLAHE, and Gaussian smoothing
│   └── fusion.py                    # Multi-operator weighted gradient fusion
│
├── measurement/                     # Geometric & Structural Damage Quantification
│   ├── __init__.py
│   ├── width.py                     # Distance transform & PCA orientation angle via covariance matrix
│   ├── length.py                    # Medial skeleton arc-length measurement
│   ├── area.py                      # Damaged surface area percentage
│   └── calibration.py               # Color thresholding & marker scale factor (px/cm)
│
├── gui/                             # User Interfaces
│   ├── app_pyside.py                # PySide6 Cyber-Nordic dark GUI (Loupe lens, 3D relief, Split view)
│   └── app.py                       # Lightweight Tkinter GUI
│
├── evaluation/                      # Validation & Metrics
│   ├── comparison.py                # Batch evaluation across 5 algorithms
│   ├── metrics.py                   # Precision, Recall, F1-Score, Accuracy
│   └── test_ai_integration.py       # 6-case automated test suite
│
├── models/
│   └── wall_crack_mobilenetv2.keras # Pre-trained MobileNetV2 crack classifier
│
├── dataset/
│   ├── raw/                         # Campus wall survey photos
│   └── ground_truth/                # Hand-drawn & verified ground truth masks
│
├── reports/                         # Generated CSV metrics & Matplotlib comparison graphs
├── main.py                          # Unified orchestrator CLI
└── requirements.txt                 # Dependencies (pure NumPy, SciPy, Pillow, PySide6, TensorFlow)
```
