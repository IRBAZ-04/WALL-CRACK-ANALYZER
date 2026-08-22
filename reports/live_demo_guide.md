# 10-MINUTE LIVE DEMONSTRATION & FACULTY VIVA GUIDE

## AUTOMATIC WALL CRACK DETECTOR & ANALYZER — UCSC313L

This step-by-step guide is designed to help you present a flawless 10-minute live project defense to your faculty evaluators.

---

### DEMO OVERVIEW & TIME ALLOCATION

```
[00:00 - 01:00]  Minute 1: Problem Statement & Objectives
[01:00 - 02:00]  Minute 2: Live Image Capture / Upload Demonstration
[02:00 - 04:00]  Minutes 3-4: Classical Preprocessing & Edge Operator Comparison
[04:00 - 06:00]  Minutes 5-6: Crack Segmentation, Morphology & Skeletonization
[06:00 - 08:00]  Minutes 7-8: Quantitative Measurements & Damage Assessment HUD
[08:00 - 09:00]  Minute 9: Benchmark Evaluation & Metric Charts
[09:00 - 10:00]  Minute 10: Conclusion, Physical Limitations & Q&A
```

---

### MINUTE-BY-MINUTE SCRIPT & ACTIONS

#### MINUTE 0:00 – 1:00 | Introduction & Problem Statement
* **Action**: Launch the PySide6 Desktop GUI (`gui/app_pyside.py`) or navigate to the **Home & Controls** tab.
* **What to say**:
  > *"Good morning respected faculty members. Today I present my UCSC313L project titled **Automatic Wall Crack Detection, Measurement, and Damage Assessment Using Classical Image Processing**.*
  > *Cracks in concrete walls and academic infrastructure are early indicators of structural degradation. Manual visual inspection is subjective, time-consuming, and lacks geometric precision. Our objective is to automate crack detection, measure physical length and width, estimate wall damage percentage, and benchmark four classical edge detection operators (Sobel, Prewitt, Laplacian, and Canny) under diverse campus lighting conditions."*

---

#### MINUTE 1:00 – 2:00 | Live Image Capture / Upload
* **Action**:
  1. Click **📂 Upload Wall Photo** to select a campus photo (e.g. `CAMPUS_ACADEMIC_001.jpg`).
  2. OR click **🎥 Live Campus Camera** to demonstrate live webcam acquisition directly inside the app.
* **What to say**:
  > *"Our application accepts high-resolution wall photographs taken around campus across daylight, strong noon sunlight, low-light evening, and shadowed conditions. Here we load a campus wall photograph featuring a structural crack and a 10 cm bright reference marker used for physical scale calibration."*

---

#### MINUTE 2:00 – 4:00 | Pipeline Preprocessing & Edge Detection Comparison
* **Action**:
  1. Select **Canny** operator from the dropdown, then click **⚡ Run Complete Pipeline**.
  2. Switch to the **🔬 Processing Stages** tab and click the **2-Up Split Comparison Slider** to drag between the Original Wall Photo and the Edge Map.
* **What to say**:
  > *"The system first resizes the image to 800 pixels width to standardize scale. Next, it converts RGB to grayscale and measures local brightness variance to classify the lighting condition. It applies CLAHE (Contrast Limited Adaptive Histogram Equalization) with parameter presets tailored to dark, bright, or shadowed environments, followed by Gaussian noise reduction.*
  > *We then run the four edge operators independently: Sobel and Prewitt compute first-order intensity gradients, Laplacian identifies second-derivative zero-crossings, and Canny performs multi-stage hysteresis tracking. Our multi-operator Fusion combines Canny details with Sobel/Prewitt gradients."*

---

#### MINUTE 4:00 – 6:00 | Crack Segmentation, Morphology & Skeletonization
* **Action**: Select the **14-Stage Pipeline Grid** tab to display intermediate binarized and morphologically cleaned stages.
* **What to say**:
  > *"For segmentation, the system automatically chooses between Otsu global thresholding for evenly-lit walls and Adaptive Gaussian thresholding for shadowed walls. Morphological opening erases surface noise specks, while morphological closing bridges hairline discontinuities.*
  > *To measure crack length accurately without measuring blob width, we perform Medial Axis Skeletonization using the Zhang-Suen algorithm to reduce the crack to a 1-pixel-wide topological centerline."*

---

#### MINUTE 6:00 – 8:00 | Measurements & Damage Quantification HUD
* **Action**: Click the **📊 Measurements HUD** tab to show the animated KPI cards and detailed breakdown panel.
* **What to say**:
  > *"Here on the Measurements HUD, the system displays real-time quantitative metrics:*
  > 1. **Detection Verdict & Confidence Score** derived from edge density and component elongation.
  > 2. **Crack Length**: Computed via geodesic path traversal along the skeleton centerline ($41.5\text{ cm}$).
  > 3. **Crack Width**: Derived using the Euclidean Distance Transform (EDT) sampled at every skeleton pixel, providing Minimum ($0.02\text{ cm}$), Maximum ($0.81\text{ cm}$), and Average ($0.19\text{ cm}$) width.
  > 4. **Damaged Wall Area Percentage**: Formulated as $\text{Damage \%} = \frac{\text{Crack Pixels}}{\text{Wall Pixels}} \times 100$, yielding $5.38\%$ surface damage."*

---

#### MINUTE 8:00 – 9:00 | Algorithm Benchmark & Performance Charts
* **Action**: Click the **📈 Algorithm Benchmark** tab to present the Precision, Recall, F1-Score, and Execution Time evaluation table and embedded Matplotlib bar chart.
* **What to say**:
  > *"To meet the faculty requirement of objective comparison without data fabrication, we evaluated all algorithms against ground-truth masks annotated across campus images:*
  > * Sobel achieved an F1-Score of **0.2052** with 0.0145s processing time.
  > * Prewitt achieved an F1-Score of **0.2051** with 0.0079s processing time.
  > * Canny achieved high localization precision and fast execution (**0.0045s**).
  > * Sobel and Prewitt provided the best balance of recall and noise suppression on rough masonry surfaces."*

---

#### MINUTE 9:00 – 10:00 | Conclusion, Physical Limitations & Q&A
* **Action**: Switch to the **✏️ Ground Truth Tool** or **🎥 Live Demo Guide** tab for final closing remarks.
* **What to say**:
  > *"In conclusion, our application provides an end-to-end engineering solution from raw image acquisition to physical damage reporting. We explicitly acknowledge real-world limitations: physical centimeter measurements rely on a planar reference marker, and extremely faint hairline cracks under heavy shadows require tuned thresholding.*
  > *Thank you. I am ready for your questions."*

---

### POTENTIAL FACULTY QUESTIONS & DEFENSE ANSWERS

* **Q1: Why did you use classical image processing instead of Deep Learning (YOLO/CNN)?**
  * **Answer**: *"Classical image processing ensures complete mathematical transparency, requires zero GPU overhead, runs instantly on standard laptops, and satisfies all UCSC313L course requirements for explicit algorithmic comparison."*

* **Q2: How is crack width calculated?**
  * **Answer**: *"We apply the Euclidean Distance Transform (EDT) on the binary crack mask. For every point on the 1-pixel skeleton centerline, the distance to the nearest background pixel represents half the crack width. Doubling this distance yields exact point-by-point width."*

* **Q3: How do you handle physical calibration?**
  * **Answer**: *"We detect a known 10 cm reference marker in the image plane to compute the pixel-to-centimeter ratio ($\text{cm/px}$). If no marker is present, the app strictly reports measurements in pixels without inventing physical dimensions."*
