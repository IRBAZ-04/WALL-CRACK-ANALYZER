# ACADEMIC PROJECT REPORT

## AUTOMATIC WALL CRACK DETECTION, MEASUREMENT AND DAMAGE ASSESSMENT USING CLASSICAL IMAGE PROCESSING

**Course Code**: UCSC313L — Image Processing Project  
**Degree Program**: Bachelor of Computer Applications (BCA)  
**Academic Year**: 2025–2026  

---

### TABLE OF CONTENTS
1. Introduction
2. Problem Statement
3. Motivation & Structural Safety Context
4. Project Objectives
5. Literature Survey & Related Work
6. Existing System Limitations
7. Proposed System Architecture
8. Campus Dataset Collection Strategy
9. Dataset Description & Categorization
10. Image Preprocessing Pipeline (Resize, CLAHE & Gaussian Denoising)
11. Sobel Edge Detection Operator
12. Prewitt Edge Detection Operator
13. Laplacian of Gaussian (LoG) Operator
14. Multi-Stage Canny Edge Detection Operator
15. Multi-Operator Edge Map Fusion
16. Crack Segmentation (Otsu & Adaptive Binarization)
17. Morphological Noise Cleaning (Opening & Closing)
18. Medial Axis Skeletonization (Zhang-Suen)
19. Crack Measurement (Length & Distance Transform Width)
20. Damage Area Calculation & Wall Surface Percentage
21. Physical Calibration via 10 cm Reference Markers
22. Ground-Truth Mask Creation & Pixel-Level Metrics
23. Experimental Results & Algorithm Comparison
24. PySide6 Desktop GUI Application Design
25. Live Campus Demonstration Protocol
26. Limitations & Future Scope
27. Conclusion
28. References

---

### CHAPTER 1: INTRODUCTION
Civil infrastructure and academic buildings undergo continuous structural wear caused by thermal expansion, foundation settlement, seismic activity, and environmental weathering. Cracks occurring in concrete and masonry walls represent the earliest visible indicator of structural deterioration. Traditional inspection relies on manual visual examination, which is time-consuming, prone to human error, subjective, and difficult to standardize.

This project presents an end-to-end automated image-processing system built with OpenCV and Python for wall crack detection, geometric quantification (length, min/max/avg width, surface area), damage severity assessment, and objective multi-algorithm comparison (Sobel vs. Prewitt vs. Laplacian vs. Canny vs. Multi-Operator Fusion).

---

### CHAPTER 2: PROBLEM STATEMENT
Manual wall crack inspection across university campuses, residential hostels, and concrete infrastructure lacks objectivity and quantitative scaling. Existing automated systems often rely on black-box deep learning models requiring high computational power or produce uncalibrated pixel outputs without physical unit conversion.

The objective of this project is to develop a classical image-processing pipeline and a modern PySide6 desktop GUI capable of:
1. Processing wall photographs under diverse real-world lighting conditions (daylight, strong noon sunlight, low-light evening, shadows, uneven illumination).
2. Segmenting crack structures from textured wall backgrounds.
3. Estimating physical crack dimensions (length in cm, width in cm/mm, damaged area percentage).
4. Evaluating four classical edge-detection operators against human-annotated ground-truth masks without data fabrication.

---

### CHAPTER 3: MOTIVATION & STRUCTURAL SAFETY CONTEXT
Early detection of micro-cracks prevents catastrophic structural failures in concrete structures. Providing maintenance engineers and campus facility administrators with an accessible desktop application enables rapid triage and periodic structural health monitoring.

---

### CHAPTER 4: PROJECT OBJECTIVES
* Implement a 14-stage classical image processing pipeline without deep learning overhead.
* Compare Sobel, Prewitt, Laplacian, and Canny edge detection algorithms independently on identical campus wall images.
* Calculate quantitative evaluation metrics: Precision, Recall, F1-Score, Jaccard Index (IoU), Pixel Accuracy, and Execution Time.
* Develop a reference marker calibration technique (10 cm physical scale).
* Provide a high-performance PySide6 desktop GUI with live webcam capture, side-by-side comparison slider, HUD metrics dashboard, and built-in ground-truth annotation painter.

---

### CHAPTER 5: LITERATURE SURVEY
Classical edge detection operators rely on spatial gradient convolution kernels:
* **Sobel (1968)**: Uses $3 \times 3$ smoothing-gradient kernels emphasizing central pixels.
* **Prewitt (1970)**: Employs uniform spatial kernels for directional derivative approximation.
* **Laplacian (1980)**: Second-derivative isotropic operator identifying zero-crossings.
* **Canny (1986)**: Multi-stage optimal edge detector utilizing non-maximum suppression and hysteresis thresholding.

---

### CHAPTER 6: EXISTING SYSTEM LIMITATIONS
Existing visual inspection methods suffer from:
* High subjectivity and lack of reproducible measurement standards.
* Failure to compensate for non-uniform lighting and shadows.
* Inability to convert pixel dimensions to physical centimeters without reference objects.

---

### CHAPTER 7: PROPOSED SYSTEM ARCHITECTURE
The system operates as a sequential pipeline:

```
[Raw Image] -> [Resize & Grayscale] -> [Lighting Classifier] -> [Adaptive CLAHE]
  -> [Gaussian Denoising] -> [Parallel Edge Detection: Sobel | Prewitt | Laplacian | Canny]
  -> [Multi-Operator Fusion] -> [Otsu / Adaptive Segmentation] -> [Morphology Filter]
  -> [Zhang-Suen Skeletonization] -> [Distance Transform Width & Path Length]
  -> [Physical Scale Calibration] -> [PySide6 GUI HUD Display]
```

---

### CHAPTER 8: CAMPUS DATASET COLLECTION STRATEGY
A target dataset of 200–250 self-collected campus images was organized across diverse locations:
* Academic Blocks (A & B)
* Student Hostels
* Staircases & Corridors
* Underground Parking Pillars
* Outer Boundary Walls

Images were captured under 4 distinct lighting environments:
1. Morning / Daylight (even diffuse light)
2. Noon / Strong Direct Sunlight (high contrast & specular reflections)
3. Low Light / Evening (artificial fluorescent illumination)
4. Shadows & Uneven Illumination (complex wall gradients)

---

### CHAPTER 9: DATASET DESCRIPTION & CATEGORIZATION
| Crack Category | Description | Count Target |
| :--- | :--- | :--- |
| Hairline Cracks | Thin width (<1.5 mm), low contrast | 40 |
| Deep / Wide Cracks | Heavy structural fracture (>5 mm width) | 40 |
| Vertical Cracks | Load-bearing orientation | 30 |
| Horizontal Cracks | Thermal or expansion joint cracks | 30 |
| Diagonal / Irregular | Shear stress fractures | 25 |
| Multiple / Branched | Intersecting crack network | 25 |
| Non-Crack Walls | Control group (textured sound concrete) | 40 |

---

### CHAPTER 10: IMAGE PREPROCESSING PIPELINE
1. **Resizing**: Standardized to width $W = 800\text{ px}$ maintaining aspect ratio.
2. **Grayscale Conversion**:
   $$Y = 0.299R + 0.587G + 0.114B$$
3. **Adaptive CLAHE**:
   Contrast Limited Adaptive Histogram Equalization with clip limit $C \in [1.5, 3.5]$ derived from local grid variance.
4. **Gaussian Denoising**:
   $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

---

### CHAPTER 11: SOBEL EDGE DETECTION OPERATOR
Convolves image $I$ with horizontal $S_x$ and vertical $S_y$ kernels:
$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
Magnitude: $M(x,y) = \sqrt{S_x^2 + S_y^2}$.

---

### CHAPTER 12: PREWITT EDGE DETECTION OPERATOR
Directional kernels with uniform weights:
$$P_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}, \quad P_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$

---

### CHAPTER 13: LAPLACIAN OF GAUSSIAN (LoG) OPERATOR
Second-derivative isotropic kernel:
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
Identifies edge locations via zero-crossings.

---

### CHAPTER 14: MULTI-STAGE CANNY OPERATOR
1. Gaussian smoothing.
2. Gradient magnitude & direction angle $\theta = \arctan(G_y / G_x)$.
3. Non-maximum suppression along gradient normal.
4. Hysteresis double thresholding ($T_{\text{low}}, T_{\text{high}}$) auto-derived from Otsu intensity level.

---

### CHAPTER 15: MULTI-OPERATOR EDGE MAP FUSION
Combines structural detail of Canny with intensity gradients of Sobel and Prewitt:
$$F(x,y) = \min(255, w_1 \cdot \text{Canny} + w_2 \cdot \text{Sobel} + w_3 \cdot \text{Prewitt})$$

---

### CHAPTER 16: CRACK SEGMENTATION
* **Otsu Global Binarization**: Minimizes intra-class variance $\sigma_w^2(t)$.
* **Adaptive Gaussian Binarization**: Computes local threshold over $21 \times 21$ block for shadowed walls.

---

### CHAPTER 17: MORPHOLOGICAL NOISE CLEANING
* **Opening**: $B \circ S = (B \ominus S) \oplus S$ to erase small surface specks.
* **Closing**: $B \bullet S = (B \oplus S) \ominus S$ to bridge hairline discontinuities.

---

### CHAPTER 18: MEDIAL AXIS SKELETONIZATION
Reduces binary crack blob to 1-pixel wide centerline using Zhang-Suen thin algorithm, preserving topology and connectedness.

---

### CHAPTER 19: CRACK MEASUREMENT & ORIENTATION ANALYSIS
1. **Length**: Sums skeleton pixels with Euclidean step weighting ($1.0$ for orthogonal, $\sqrt{2}$ for diagonal steps).
2. **Width**: Evaluates Euclidean Distance Transform (EDT) values along skeleton centerline:
   $$W_{\text{local}} = 2 \cdot \text{EDT}(x, y)$$
   Computes minimum, maximum, and average width.
3. **Crack Orientation & PCA Direction**:
   Computes the principal axis angle $\theta$ of skeleton coordinates via Image Central Moments / Principal Component Analysis (PCA):
   $$\theta = \arctan\left(\frac{v_y}{v_x}\right)$$
   * $\theta \ge 70^\circ \rightarrow$ **Vertical Load-Bearing Shear Crack**
   * $\theta \le 20^\circ \rightarrow$ **Horizontal Expansion / Settlement Joint**
   * $20^\circ < \theta < 70^\circ \rightarrow$ **Diagonal Structural Shear Crack**
4. **Structural Severity Triage System**:
   Categorizes risk into **LOW** ($<1.0\text{ mm}$ width, $<1.5\%$ damage), **MEDIUM** ($1-3\text{ mm}$ width), **HIGH** ($3-6\text{ mm}$ width), or **CRITICAL** ($>6\text{ mm}$ width or $>8\%$ damage).
5. **Color-Coded Pseudocolor Heatmap Overlay**:
   Applies OpenCV Jet colormap ($\text{COLORMAP\_JET}$) over distance-transform normalized values to render a visual width intensity map.

---

### CHAPTER 20: DAMAGE AREA CALCULATION & WALL PERCENTAGE
$$\text{Damage Percentage} = \left(\frac{\text{Crack Pixels}}{\text{Total Wall Pixels}}\right) \times 100$$

---

### CHAPTER 21: PHYSICAL CALIBRATION VIA 10 CM MARKER
Using a known $10\text{ cm}$ reference object in the image plane:
$$\text{Scale } (\text{cm/px}) = \frac{10\text{ cm}}{\text{Marker Width (px)}}$$
Physical length and area are reported in $\text{cm}$ and $\text{cm}^2$.

---

### CHAPTER 22: GROUND-TRUTH MASK CREATION & PIXEL-LEVEL METRICS
Ground-truth binary masks $GT$ were generated for dataset evaluation.
* $\text{Precision} = \frac{TP}{TP + FP}$
* $\text{Recall} = \frac{TP}{TP + FN}$
* $\text{F1-Score} = \frac{2 \cdot P \cdot R}{P + R}$
* $\text{IoU (Jaccard)} = \frac{TP}{TP + FP + FN}$

---

### CHAPTER 23: EXPERIMENTAL RESULTS & ALGORITHM COMPARISON
Experimental performance results on campus dataset:

| Algorithm | Precision | Recall | F1-Score | Accuracy | Execution Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Canny** | 0.1380 | 0.7635 | 0.1494 | 0.3016 | **0.0045s** |
| **Sobel** | 0.1837 | 0.7988 | **0.2052** | 0.2918 | 0.0145s |
| **Prewitt** | **0.1831** | 0.8080 | 0.2051 | **0.3005** | 0.0079s |
| **Laplacian**| 0.0067 | **0.8703**| 0.0132 | 0.0668 | 0.0107s |
| **Fusion** | 0.1193 | 0.7877 | 0.1394 | 0.2995 | 0.0099s |

---

### CHAPTER 24: PYSIDE6 DESKTOP GUI DESIGN
The PySide6 interface provides:
* Nordic Dark Glassmorphic QSS visual styling.
* **Interactive Magnifying Glass Loupe Lens (Hover Inspection Tool)**: As mouse cursor moves over the image, a circular $2.5\times$ magnified lens pops up with cyan reticle crosshairs for pixel-level edge inspection.
* 2-Up Side-by-Side Split Image Comparison slider with synchronous dragging.
* Live webcam frame acquisition dialog.
* HUD dashboard card metrics.
* In-app Ground Truth Mask Painter tool.
* Embedded Matplotlib benchmark charts.

---

### CHAPTER 25: LIVE CAMPUS DEMONSTRATION PROTOCOL
A 10-minute structured presentation sequence designed for project viva defense.

---

### CHAPTER 26: LIMITATIONS & FUTURE SCOPE
* Physical calibration requires reference marker in the image plane.
* Highly textured masonry brick patterns require fine-tuned morphological kernel sizes.
* Future work includes integrating stereo vision for 3D crack depth profiling.

---

### CHAPTER 27: CONCLUSION
This project successfully developed a classical image processing pipeline and desktop GUI for automatic wall crack detection, measurement, damage assessment, and algorithm benchmarking, meeting all UCSC313L course requirements.

---

### CHAPTER 28: REFERENCES
1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Canny, J. (1986). A computational approach to edge detection. *IEEE TPAMI*, 8(6), 679-698.
3. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE TSMC*, 9(1), 62-66.
