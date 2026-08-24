# Automatic Wall Crack Detection, Measurement and Damage Assessment

**Course**: UCSC313L Image Processing Project (BCA)  
**Implementation**: 100% Classical Image Processing (OpenCV + PySide6 Desktop GUI) — No Deep Learning.

---

## Key Features & Novelty

1. **Interactive Magnifying Glass Loupe Lens (Hover Inspection Tool)**:
   * As you move your mouse cursor over the wall image in the Desktop GUI, a circular $2.5\times$ magnified loupe lens pops up with reticle crosshairs for pixel-level edge and crack inspection.
2. **🏔️ 3D Structural Elevation & Fracture Relief Landscape**:
   * Renders a 3D surface mesh plot (using Matplotlib 3D) where sound wall surfaces form high plateaus while crack fractures carve out deep structural ravines and canyons.
3. **Lighting-Aware Adaptive Pipeline**:
   * `preprocessing.py` classifies each image's lighting condition (`daylight`, `noon_sunlight`, `low_light_evening`, `shadows_uneven`) from histogram statistics and automatically picks CLAHE parameters; `segmentation.py` automatically switches between Otsu and Adaptive Gaussian thresholding.
4. **4-Algorithm Comparison + Multi-Operator Fusion**:
   * Compares **Sobel, Prewitt, Laplacian, and Canny** algorithms independently, along with a weighted multi-operator **Fusion** method.
5. **Structural Risk Rating Triage System**:
   * Evaluates crack width and damage % to assign color-coded severity levels: **LOW**, **MEDIUM**, **HIGH**, and **CRITICAL**.
6. **Mathematical Crack Orientation ($\theta$ Angle via PCA)**:
   * Computes principal axis orientation to classify cracks into **Vertical Load-Bearing**, **Horizontal Joint/Settlement**, or **Diagonal Shear** fractures.
7. **2-Up Side-by-Side Split Image Comparison Slider**:
   * Drag handle to compare original campus wall photos against processed crack edge overlays in real time.
8. **In-App Ground Truth Annotation Painter**:
   * Draw ground-truth crack masks directly on campus photos inside the PySide6 app using a brush tool to populate evaluation ground truth data.

---

## Setup & Running Instructions

### 1. Install Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Launch PySide6 Desktop GUI
```powershell
.\.venv\Scripts\python.exe gui/app_pyside.py
```

### 3. Single Image Pipeline CLI
```powershell
.\.venv\Scripts\python.exe main.py dataset/raw/CAMPUS_ACADEMIC_001.jpg canny
```

### 4. Run Batch Algorithm Evaluation
```powershell
.\.venv\Scripts\python.exe evaluation/comparison.py
```

---

## Academic Deliverables Included
* **[20–25 Page Academic Project Report](file:///d:/SEM5%20PROJECTS/WALL%20CRACK%20ANALYZER/reports/academic_project_report.md)** (`reports/academic_project_report.md`)
* **[10-Minute Live Demo Presentation Guide](file:///d:/SEM5%20PROJECTS/WALL%20CRACK%20ANALYZER/reports/live_demo_guide.md)** (`reports/live_demo_guide.md`)
