# 🛰️ S.E.N.T.I.N.E.L.
**Strategic Environment & Neural Terrain Intelligence Logic**

[![Live Deployment](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Ashirwad12/SENTINEL)
[![mAP Score](https://img.shields.io/badge/mAP@50-0.715-success)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](#)

S.E.N.T.I.N.E.L. is an advanced, physics-grounded computer vision system designed to provide commanders with a visually accurate simulation of kinetic strikes on aerial reconnaissance imagery[cite: 1]. Developed for the IIT BHU "Serve Smart" Hackathon, this platform eliminates the need for manual scale calibration by dynamically analyzing detected military assets[cite: 1].

**Try the live dashboard here:** [S.E.N.T.I.N.E.L. Strike Command Interface](https://huggingface.co/spaces/Ashirwad12/SENTINEL)

---

## 🧠 System Architecture

The platform operates across four distinct logic layers[cite: 1]:
1. **Perception Layer:** Utilizes a custom-trained YOLOv8-Medium model to detect military and civilian assets[cite: 1].
2. **Calibration Layer (Global Scale Engine):** Establishes the real-world scale of the image by scanning detections for the most reliable "Reference Object"[cite: 1].
3. **Simulation Layer:** Computes optimal strike coordinates and references a verified weapon database for blast radii[cite: 1].
4. **Visualization Layer:** Renders the Heads-Up Display (HUD) using OpenCV, mapping out "Kill Zones" and shockwaves to scale[cite: 1].

---

## 🧮 The Mathematical Core

S.E.N.T.I.N.E.L. relies on a novel "Hierarchy of Rulers" approach to understand battlefield dimensions without metadata[cite: 1]. 

### 1. Auto-Calibration (The Global Scale Factor)
The system trusts larger, standard rigid bodies over variable organic objects[cite: 1]. The priority ranking is[cite: 1]:
> Warship > Aircraft > Tank > Truck > Soldier

Once the highest-priority reference object $(Obj_{ref})$ is identified, the system calculates the global Pixels-Per-Meter ratio $(S_{global})$[cite: 1]:

$$S_{global}=\frac{L_{pixel}}{L_{real}}$$

Where $L_{pixel}$ is the maximum bounding box dimension in pixels, and $L_{real}$ is the verified database length of the object in meters (e.g., Aircraft = $18.0\text{m}$)[cite: 1].

### 2. Physics-Based Blast Rendering
Blast radii are not arbitrary; they are hard-coded to military specifications[cite: 1]. To draw the weapon impact circle on the HUD[cite: 1]:

$$R_{draw}=R_{weapon}\times S_{global}$$

*Example:* A tactical nuclear strike has a kill radius ($R_{weapon}$) of $600\text{m}$[cite: 1]. The visual diameter drawn on screen will be exactly double ($1200\text{m}$) multiplied by the scale factor[cite: 1].

### 3. Weighted Center of Mass Targeting
Impact coordinates $(I_{x}, I_{y})$ are calculated based on threat density rather than random targeting[cite: 1]. The system uses the following weighted algorithms[cite: 1]:

$$I_{x}=\frac{\sum(x_{i}\cdot W_{i})}{\sum W_{i}} \quad \text{and} \quad I_{y}=\frac{\sum(y_{i}\cdot W_{i})}{\sum W_{i}}$$

Where $W_{i}$ represents the specific threat weight[cite: 1]:
*   **Warship:** $15.0$[cite: 1]
*   **Tank:** $10.0$[cite: 1]
*   **Soldier:** $2.0$[cite: 1]
*   **Civilian:** $0.0$ (Explicitly ignored in targeting computations)[cite: 1]

---

## 🎯 Model Training & AI Optimization

The underlying detection engine is a **YOLOv8-Medium** model trained at a high resolution of $1024\text{px}$ to overcome the "Small Object" constraint common in aerial imagery[cite: 2].

*   **Performance:** Achieved **$0.715\text{ mAP@50}$** (Run_HighRes_Leg4)[cite: 2].
*   **Data Hygiene:** Severely imbalanced and rare classes (such as Civilian and Trench) were actively removed from training to prevent negative transfer and hallucinated detections[cite: 2].
*   **Cyclic Relay Race Strategy:** The model was trained in controlled "Legs," allowing for manual checkpointing, hyperparameter tuning, and learning rate resets to escape local minima[cite: 2].
*   **Inference Stability:** We utilize Test-Time Augmentation (TTA) and a "Rosetta Stone" reverse-mapping protocol to restore official class IDs during post-processing[cite: 2].
*   **Efficiency:** The final `.pt` weights are approximately $50\text{ MB}$, allowing for rapid CPU-edge deployment without sacrificing accuracy[cite: 2].

---

## 📂 Repository Structure

```text
SENTINEL/
│
├── docs/                                   # Project Documentation
│   ├── SENTINEL_Technical_Dossier.pdf      # The logic and math breakdown
│   └── High_Res_Detection_Report.pdf       # The YOLOv8 training methodology
│
├── notebooks/                              # The Core Pipeline Notebooks
│   ├── 01_EDA.ipynb                        # Data Hygiene analysis
│   ├── 02_Training.ipynb                   # Cyclic YOLOv8 training
│   ├── 03_Inference.ipynb                  # Test-Time Augmentation
│   ├── Sentinel_Colabimg.ipynb             # MGRS & Coordinate Logic
│   └── video_collab_notebook.ipynb         # Video HUD integration
│
├── assets/Report_Assets/                   # Visual Data Outputs
│   ├── class_distribution.png              
│   ├── image_sizes.png                     
│   ├── object_sizes.png                    
│   └── sample_visualization.png            
│
├── app.py                                  # The Main S.E.N.T.I.N.E.L. Dashboard 
├── requirements.txt                        # Dependencies
└── README.md                               # Project overview
