import os
import gc
import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="S.E.N.T.I.N.E.L. // STRIKE COMMAND",
    layout="wide",
    page_icon="☢️",
    initial_sidebar_state="expanded"
)

# FORCE MEMORY FLUSH
gc.collect()

# CRITICAL MEMORY SAFEGUARDS
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

MODEL_PATH = "model.pt"

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# ==========================================
# 2. INTELLIGENCE DATABASES
# ==========================================

# WEAPON SPECIFICATIONS (RADIUS IN METERS)
# NOTE: Visual Diameter will be 2x Radius
WEAPON_SYSTEMS = {
    "ARTILLERY (155mm Excalibur)": { 
        "type": "Precision", 
        "radius_meters": 12,  # 24m Diameter (~1.3x Aircraft)
        "desc": "POINT TARGET NEUTRALIZATION", 
        "color": (0, 255, 0) 
    },
    "CRUISE MISSILE (Tomahawk)": { 
        "type": "Area", 
        "radius_meters": 45,  # 90m Diameter (~5.0x Aircraft)
        "desc": "HEAVY INFRASTRUCTURE DESTRUCTION", 
        "color": (0, 200, 255) 
    },
    "HYPERSONIC (Avangard HGV)": { 
        "type": "Area", 
        "radius_meters": 75,  # 150m Diameter
        "desc": "KINETIC SHOCKWAVE", 
        "color": (0, 100, 255) 
    },
    "BALLISTIC MISSILE (Jericho)": { 
        "type": "Strategic", 
        "radius_meters": 120, # 240m Diameter
        "desc": "BASE LEVEL DESTRUCTION", 
        "color": (0, 0, 255) 
    },
    "NUCLEAR (Tactical Yield)": { 
        "type": "Doomsday", 
        "radius_meters": 600, # 1200m Diameter
        "desc": "SECTOR ANNIHILATION", 
        "color": (0, 0, 180) 
    }
}

# HIERARCHY OF RULERS (Priority List)
PRIORITY_RANKING = {
    'WARSHIP': 10,   # Best (Huge & Standard)
    'AIRCRAFT': 9,   # Very Good
    'TANK': 8,       # GOLD STANDARD
    'TRUCK': 7,      # Reliable
    'ARTILLERY': 6,  # Okay
    'VEHICLE': 5,    # Variable (Car vs SUV)
    'SOLDIER': 1,    # Last Resort (Prone vs Standing)
    'WEAPON': 0      # Unreliable
}

# REAL WORLD SIZES (Meters)
# Updated Aircraft to 18.0m to match your specification
REAL_WORLD_SIZES = {
    "SOLDIER": 1.7, 
    "CIVILIAN": 1.7, 
    "WEAPON": 1.0, 
    "VEHICLE": 4.5, 
    "TRUCK": 7.5, 
    "TANK": 10.5,     
    "ARTILLERY": 10.0, 
    "AIRCRAFT": 18.0, # FIXED: 18m
    "WARSHIP": 200.0, 
    "CAMO": 6.0
}

REVERSE_MAP = { 0:0, 1:1, 2:2, 3:3, 4:4, 5:6, 6:8, 7:10, 8:11 }
LABELS = { 0:'CAMO', 1:'WEAPON', 2:'TANK', 3:'TRUCK', 4:'VEHICLE', 5:'CIVILIAN', 6:'SOLDIER', 8:'ARTILLERY', 10:'AIRCRAFT', 11:'WARSHIP' }
THREAT_WEIGHTS = { 11: 15.0, 2:10.0, 8:9.0, 10:8.0, 3:6.0, 1:5.0, 0:2.0, 6:2.0, 5:0.0 }

# ==========================================
# 3. UI THEME
# ==========================================
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&display=swap" rel="stylesheet">
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?q=80&w=3840&auto=format&fit=crop");
        background-size: cover; background-position: center; background-attachment: fixed;
        color: #e2e8f0; font-family: 'Rajdhani', sans-serif;
    }
    h1, h2, h3 { 
        font-family: 'Orbitron', sans-serif !important; 
        color: #00f0ff !important; 
        text-shadow: 0 0 15px rgba(0, 240, 255, 0.6); 
    }
    .glass-panel { 
        background: rgba(10, 15, 25, 0.85); 
        border: 1px solid #00f0ff; 
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.1);
        backdrop-filter: blur(10px); 
        border-radius: 6px; 
        padding: 20px; 
        margin-bottom: 20px; 
    }
    .weapon-card { 
        border-left: 4px solid #ff3333; 
        background: linear-gradient(90deg, rgba(80,0,0,0.6) 0%, rgba(0,0,0,0) 100%); 
        padding: 12px; 
        margin-bottom: 15px; 
        border-radius: 4px; 
        font-family: 'Orbitron', sans-serif;
    }
    .target-row { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background: rgba(0, 30, 50, 0.5); 
        border-left: 3px solid #00f0ff; 
        margin-bottom: 6px; 
        padding: 10px; 
        border-radius: 2px; 
    }
    [data-testid="stSidebar"] { 
        background-color: rgba(5, 8, 12, 0.92); 
        border-right: 1px solid #004444; 
        backdrop-filter: blur(8px); 
    }
    .stButton button { 
        background-color: rgba(200, 0, 0, 0.6); 
        color: white; 
        border: 1px solid red; 
        font-family: 'Orbitron', sans-serif; 
        width: 100%; 
        transition: all 0.3s;
    }
    .stButton button:hover { 
        background-color: rgba(255, 0, 0, 1.0); 
        border: 1px solid white; 
        box-shadow: 0 0 15px rgba(255, 0, 0, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. CORE FUNCTIONS
# ==========================================

@st.cache_resource(max_entries=1)
def load_model():
    return YOLO(MODEL_PATH)

def safe_resize(image, max_width=1280): 
    w, h = image.size
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        return image.resize((max_width, new_h), Image.Resampling.LANCZOS)
    return image

def load_image(file):
    img = Image.open(file)
    return safe_resize(img)

def get_grid_ref(x, y, w, h, cls_id):
    e = int((x / w) * 10000)
    n = int((y / h) * 10000)
    if cls_id == 11: return f"NAV-SEC {e:04d}-{n:04d} [WATER]"
    return f"34S WC {e:04d} {n:04d}"

def calculate_impact_point(detections, weapon_type):
    if not detections: return None
    
    # Precision: Hit highest threat
    if weapon_type == "Precision":
        best_target = max(detections, key=lambda d: THREAT_WEIGHTS.get(d['class'], 0))
        if THREAT_WEIGHTS.get(best_target['class'], 0) > 0: 
            return best_target['center']
    
    # Area: Hit center of mass
    total_weight, wx, wy = 0, 0, 0
    for det in detections:
        w = THREAT_WEIGHTS.get(det['class'], 0)
        if weapon_type == "Doomsday": w = 1.0 
        if w > 0:
            cx, cy = det['center']
            wx += cx * w
            wy += cy * w
            total_weight += w
    if total_weight > 0: return (int(wx / total_weight), int(wy / total_weight))
    return None

def get_global_best_scale(detections):
    """
    HIERARCHY OF RULERS LOGIC:
    1. Scan all objects.
    2. Pick the one with the HIGHEST Priority Rank.
    3. Use that object to set the global scale.
    """
    if not detections:
        return 1.0, "DEFAULT"
    
    best_obj = None
    best_priority = -1
    
    for det in detections:
        label = LABELS.get(det['class'], 'UNK')
        p = PRIORITY_RANKING.get(label, 0)
        
        if p > best_priority:
            best_priority = p
            best_obj = det
        elif p == best_priority:
            x1, y1, x2, y2 = det['box']
            curr_size = max(x2-x1, y2-y1)
            
            bx1, by1, bx2, by2 = best_obj['box']
            best_size = max(bx2-bx1, by2-by1)
            
            if curr_size > best_size:
                best_obj = det

    if best_obj:
        label = LABELS.get(best_obj['class'], 'UNK')
        x1, y1, x2, y2 = best_obj['box']
        pixel_size = max(x2 - x1, y2 - y1)
        real_size = REAL_WORLD_SIZES.get(label, 5.0)
        
        scale = pixel_size / real_size
        return scale, label 
        
    return 1.5, "DEFAULT"

def draw_hud(img, detections, impact_point, weapon_config, scale_factor):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 10, 20), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # DRAW TARGETS
    for i, det in enumerate(detections):
        x1, y1, bw, bh = det['box']
        label = LABELS.get(det['class'], 'UNK')
        tid = i + 1
        is_threat = label != 'CIVILIAN'
        color = (50, 50, 255) if is_threat else (255, 200, 0)
        
        l = int(min(bw, bh) / 4)
        cv2.line(img, (x1, y1), (x1+l, y1), color, 2)
        cv2.line(img, (x1, y1), (x1, y1+l), color, 2)
        cv2.line(img, (x1+bw, y1+bh), (x1+bw-l, y1+bh), color, 2)
        cv2.line(img, (x1+bw, y1+bh), (x1+bw, y1+bh-l), color, 2)
        cv2.rectangle(img, (x1, y1-22), (x1+45, y1-2), color, -1)
        cv2.putText(img, f"T-{tid:02d}", (x1+4, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    if impact_point:
        ix, iy = impact_point
        color = weapon_config["color"]
        
        # --- MULTI-ZONE SYSTEM ---
        
        # 1. INNER KILL ZONE
        kill_meters = weapon_config["radius_meters"]
        kill_px = int(kill_meters * scale_factor)
        kill_px = max(15, min(kill_px, int(w * 1.5)))
        
        # 2. OUTER SHOCKWAVE ZONE (2x Radius)
        shock_px = int(kill_px * 2.0)
        shock_px = max(30, min(shock_px, int(w * 2.0)))

        # 3. DRAWING
        cv2.circle(img, (ix, iy), shock_px, color, 1) # Outer (Faint)
        cv2.circle(img, (ix, iy), kill_px, color, 3) # Inner (Strong)
        
        # Crosshair
        cross_size = 15
        cv2.line(img, (ix-cross_size, iy), (ix+cross_size, iy), color, 2)
        cv2.line(img, (ix, iy-cross_size), (ix, iy+cross_size), color, 2)
        cv2.circle(img, (ix, iy), 3, color, -1)
        
        # Labels
        cv2.putText(img, f"KILL ({kill_meters}m)", (ix+10, iy-kill_px-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        cv2.putText(img, "BLAST", (ix+10, iy-shock_px-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img

# ==========================================
# 5. UI EXECUTION
# ==========================================

st.sidebar.title("S.E.N.T.I.N.E.L.")
st.sidebar.markdown("###  WEAPON AUTHORIZATION")

selected_weapon_name = st.sidebar.selectbox("SELECT DELIVERY SYSTEM", list(WEAPON_SYSTEMS.keys()))
weapon_config = WEAPON_SYSTEMS[selected_weapon_name]

if st.sidebar.button(" RESET MISSION / NEW UPLOAD"):
    st.session_state.uploader_key += 1
    st.rerun()

st.sidebar.markdown("###  SCALE CALIBRATION")
scale_mode = st.sidebar.radio("OPTICAL ZOOM SOURCE:", ["AUTO (AI ESTIMATED)", "MANUAL PRESET"], index=0)

manual_scale_factor = 1.0 
ref_label = "MANUAL"

if scale_mode == "MANUAL PRESET":
    preset = st.sidebar.selectbox("SELECT SENSOR ALTITUDE:", ["ORBIT (Satellite)", "HIGH ALT (Recon)", "LOW ALT (Tactical)", "CLOSE AIR (Drone)"])
    altitude_map = {"ORBIT (Satellite)": 0.2, "HIGH ALT (Recon)": 0.8, "LOW ALT (Tactical)": 1.5, "CLOSE AIR (Drone)": 3.0}
    manual_scale_factor = altitude_map[preset]

st.sidebar.markdown(f"""
<div class="weapon-card">
    <b>SYSTEM:</b> {selected_weapon_name}<br>
    <b>KILL RADIUS:</b> {weapon_config['radius_meters']}m<br>
    <b>BLAST RADIUS:</b> {weapon_config['radius_meters']*2}m
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
confidence = st.sidebar.slider("SCAN SENSITIVITY", 0.1, 1.0, 0.25)

st.title("🛰️ STRIKE COMMAND INTERFACE")
st.markdown(f"**PROTOCOL: {weapon_config['type'].upper()} // TARGETING: ACTIVE**")

uploaded_file = st.file_uploader(">> UPLOAD RECON IMAGERY", type=['jpg', 'png', 'jpeg'], key=st.session_state.uploader_key)

if uploaded_file is not None:
    image = load_image(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    st.markdown("###  TARGET ACQUISITION")
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="LIVE FEED")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner(' CALCULATING...'):
        model = load_model()
        results = model(img_bgr, conf=confidence, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            m_cls = int(box.cls[0])
            r_cls = REVERSE_MAP.get(m_cls, m_cls)
            x, y, x2, y2 = box.xywh[0].tolist()
            x1, y1 = int(x - x2/2), int(y - y2/2)
            grid_ref = get_grid_ref(x, y, w, h, m_cls)
            detections.append({'class': r_cls, 'center': (int(x), int(y)), 'box': (x1, y1, int(x2), int(y2)), 'grid': grid_ref})
    
    del results
    gc.collect()

    impact_point = calculate_impact_point(detections, weapon_config['type'])
    
    if scale_mode == "AUTO (AI ESTIMATED)":
        final_scale, ref_label = get_global_best_scale(detections)
        if ref_label == "DEFAULT":
            st.sidebar.warning(" NO REFERENCE OBJECTS. USING DEFAULT SCALE.")
        else:
            st.sidebar.success(f" CALIBRATED ON: {ref_label}")
    else:
        final_scale = manual_scale_factor

    final_img = draw_hud(img_bgr.copy(), detections, impact_point, weapon_config, final_scale)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    st.markdown("###  STRIKE SIMULATION")
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.image(final_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("###  DATA MANIFEST")
    if len(detections) > 0:
        with st.container():
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            for i, det in enumerate(detections):
                tid = i + 1
                lbl = LABELS.get(det['class'], 'UNK')
                st.markdown(f"""<div class="target-row"><span><b style="color:#ffcc00">T-{tid:02d}</b> {lbl}</span><code style="color:#00f0ff">{det['grid']}</code></div>""", unsafe_allow_html=True)
            if impact_point:
                ix, iy = impact_point
                grid_impact = get_grid_ref(ix, iy, w, h, 0)
                alert_color = '#ff0000' if weapon_config['type'] == 'Doomsday' else '#00f0ff'
                st.markdown(f"""<div style="border:2px solid {alert_color}; background:rgba(0,0,0,0.5); padding:15px; margin-top:20px; text-align:center;"><h3 style="margin:0; color:{alert_color} !important;">🚀 FINAL LAUNCH COORDINATES</h3><div style="font-size:1.8em; font-family:'Courier New'; font-weight:bold; color:#ffffff;">{grid_impact}</div><div style="color:#aaa; font-size:0.9em; margin-top:5px;">KILL RADIUS: {weapon_config['radius_meters']}M</div></div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("NO TARGETS.")
        
    del image, img_array, img_bgr, final_img, final_rgb
    gc.collect()
