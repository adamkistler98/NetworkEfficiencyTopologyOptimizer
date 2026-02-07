import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import pandas as pd
import time

# --- 1. DARK MODE & STEALTH CSS ---
st.set_page_config(
    page_title="Neuromorphic Topology V8", 
    layout="wide", 
    page_icon="🕸️",
    initial_sidebar_state="expanded"
)

# FORCE TOTAL BLACKOUT (Stealth Mode)
st.markdown("""
<style>
    /* MAIN CONTAINER */
    .stApp { background-color: #000000; color: #00E5FF; }
    
    /* SIDEBAR - FORCE BLACK */
    section[data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #111 !important;
    }
    
    /* INPUTS, DROPDOWNS & TEXT AREAS */
    .stSelectbox div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"], 
    input.st-ai, 
    textarea.st-ai,
    .stSlider div[data-baseweb="slider"] {
        color: #00E5FF !important;
    }

    /* DROPDOWN MENU ITEMS */
    ul[data-baseweb="menu"] { background-color: #000000 !important; border: 1px solid #333; }
    li[data-baseweb="option"] { color: #00E5FF !important; }
    
    /* BUTTONS */
    .stButton>button {
        color: #00E5FF !important;
        border: 1px solid #333 !important;
        background-color: #000000 !important;
        transition: all 0.3s;
        font-family: 'Courier New', monospace;
    }
    .stButton>button:hover {
        border-color: #00E5FF !important;
        box-shadow: 0 0 10px rgba(0,229,255,0.2);
        color: #FFFFFF !important;
    }

    /* TEXT & METRICS */
    h1, h2, h3, h4, p, label { color: #00E5FF !important; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 20px; }
    div[data-testid="stMetricLabel"] { color: #666 !important; font-size: 10px; }
    
    /* HIDE DEFAULT STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS ENGINE (Bio-Router) ---

class BioEngine:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.agents = np.zeros((num_agents, 3))
        # Start in random cluster
        self.agents[:, 0] = np.random.uniform(width*0.4, width*0.6, num_agents)
        self.agents[:, 1] = np.random.uniform(height*0.4, height*0.6, num_agents)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, nodes, speed, decay, sensor_angle=0.7, sensor_dist=9.0):
        self.steps += 1
        
        # 1. SENSE
        angles = self.agents[:, 2]
        def get_pos(a):
            x = (self.agents[:, 0] + np.cos(a) * sensor_dist) % self.width
            y = (self.agents[:, 1] + np.sin(a) * sensor_dist) % self.height
            return x.astype(int), y.astype(int)

        lx, ly = get_pos(angles - sensor_angle)
        cx, cy = get_pos(angles)
        rx, ry = get_pos(angles + sensor_angle)
        
        l_val = self.trail_map[ly, lx]
        c_val = self.trail_map[cy, cx]
        r_val = self.trail_map[ry, rx]
        
        # 2. DECIDE
        jitter = np.random.uniform(-0.2, 0.2, self.num_agents)
        move_fwd = (c_val > l_val) & (c_val > r_val)
        move_left = (l_val > c_val) & (l_val > r_val)
        move_right = (r_val > c_val) & (r_val > l_val)
        
        self.agents[move_left, 2] -= 0.5
        self.agents[move_right, 2] += 0.5
        self.agents[~(move_fwd | move_left | move_right), 2] += jitter[~(move_fwd | move_left | move_right)]

        # 3. MOVE
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        # 4. DEPOSIT
        ix, iy = self.agents[:, 0].astype(int), self.agents[:, 1].astype(int)
        np.add.at(self.trail_map, (iy, ix), 1.0) 
        
        # 5. ATTRACTION (Data Centers)
        for sx, sy in nodes:
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 5.0 

        # 6. DECAY
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 3. SESSION STATE ---
# Key 'engine_v8' ensures fresh start
if 'engine_v8' not in st.session_state:
    st.session_state.engine_v8 = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = [[100, 100], [200, 100], [200, 200], [100, 200], [150, 150]]
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. SIDEBAR CONTROLS (RESTORED!) ---
st.sidebar.title("🎛️ SYSTEM KERNEL")
is_running = st.sidebar.toggle("🟢 SYSTEM ONLINE", value=True)

st.sidebar.markdown("### 🗺️ TOPOLOGY PRESETS")
preset = st.sidebar.selectbox("Select Layout", ["Square Loop", "Pentagon", "Random Scatter", "Hub & Spoke"])

if st.sidebar.button("⚠️ REBOOT SIMULATION"):
    st.session_state.engine_v8 = None
    st.session_state.history = []
    if preset == "Square Loop":
        st.session_state.nodes = [[100, 100], [200, 100], [200, 200], [100, 200]]
    elif preset == "Pentagon":
        c, r = (150, 150), 80
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        st.session_state.nodes = [[c[0] + r*np.cos(a), c[1] + r*np.sin(a)] for a in angles]
    elif preset == "Random Scatter":
        st.session_state.nodes = np.random.randint(40, 260, size=(6, 2)).tolist()
    elif preset == "Hub & Spoke":
        nodes = [[150, 150]]
        nodes.extend([[150 + 100*np.cos(a), 150 + 100*np.sin(a)] for a in np.linspace(0, 2*np.pi, 7)[:-1]])
        st.session_state.nodes = nodes
    st.rerun()

st.sidebar.markdown("---")
# RESTORED PHYSICS VARIABLES
with st.sidebar.expander("⚙️ PHYSICS PARAMETERS", expanded=True):
    agent_count = st.slider("Particle Flux", 1000, 10000, 5000)
    decay_rate = st.slider("Entropy Decay", 0.90, 0.99, 0.92)
    speed_val = st.slider("Propagation C", 1.0, 5.0, 2.0)

# --- 5. INITIALIZE ENGINE ---
if st.session_state.engine_v8 is None or st.session_state.engine_v8.num_agents != agent_count:
    st.session_state.engine_v8 = BioEngine(300, 300, agent_count)

engine = st.session_state.engine_v8
nodes_arr = np.array(st.session_state.nodes)

# --- 6. MAIN LOOP ---
if is_running:
    for _ in range(10): 
        engine.step(st.session_state.nodes, speed_val, decay_rate)

# --- 7. UI & METRICS ---
st.title("NEUROMORPHIC TOPOLOGY SOLVER v8")

# METRICS
if len(nodes_arr) > 1:
    dist_mat = distance_matrix(nodes_arr, nodes_arr)
    mst_cost = minimum_spanning_tree(dist_mat).toarray().sum()
else:
    mst_cost = 0

bio_pixels = np.sum(engine.trail_map > 1.0)
bio_cost_est = bio_pixels / 8.0 

c1, c2, c3, c4 = st.columns(4)
c1.metric("NODES ONLINE", f"{len(nodes_arr)}")
c2.metric("EPOCH", f"{engine.steps}")
c3.metric("MST COST", f"{int(mst_cost)}")
c4.metric("BIO-COST", f"{int(bio_cost_est)}", delta=f"{int(mst_cost - bio_cost_est)}")

st.markdown("---")

# --- 8. DUAL GRAPHICS (RESIZED & SIDE BY SIDE) ---
col_left, col_right = st.columns(2)

# LEFT: BIO-PROCESS
with col_left:
    st.markdown("#### 🧬 BIO-PROCESS")
    # Reduced figsize from 5x5 to 3.5x3.5
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), facecolor='black')
    
    disp_map = np.log1p(engine.trail_map)
    ax1.imshow(disp_map, cmap='magma', origin='upper', vmin=0, vmax=np.percentile(disp_map, 99.5) if np.any(disp_map) else 1)
    
    if len(nodes_arr) > 0:
        ax1.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='white', s=30, edgecolors='cyan', zorder=10)
    
    ax1.axis('off')
    fig1.tight_layout(pad=0)
    st.pyplot(fig1, use_container_width=True)

# RIGHT: DIGITAL TWIN
with col_right:
    st.markdown("#### 💠 DIGITAL TWIN")
    # Reduced figsize from 5x5 to 3.5x3.5
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), facecolor='black')
    ax2.set_facecolor('black')
    ax2.set_xlim(0, 300)
    ax2.set_ylim(300, 0)
    
    y_trail, x_trail = np.where(engine.trail_map > 2.5)
    if len(x_trail) > 0:
        ax2.scatter(x_trail, y_trail, c='#00E5FF', s=0.5, alpha=0.5)
    
    if len(nodes_arr) > 0:
        ax2.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='#00E5FF', s=60, marker='s', edgecolors='white', zorder=10)

    ax2.axis('off')
    fig2.tight_layout(pad=0)
    st.pyplot(fig2, use_container_width=True)

# --- 9. TELEMETRY (FIXED) ---
st.markdown("#### 📉 CONVERGENCE TELEMETRY")

# Append new data
st.session_state.history.append({"MST": float(mst_cost), "BIO": float(bio_cost_est)})
if len(st.session_state.history) > 100: st.session_state.history.pop(0)

# Create DataFrame explicitly to prevent color mismatch errors
chart_df = pd.DataFrame(st.session_state.history)

if not chart_df.empty:
    # Ensure correct column types
    chart_df = chart_df.astype(float)
    # Streamlit line_chart auto-assigns colors if you don't map them manually, 
    # but providing a list works if columns match. 
    # To be safe, we just let it default or map specifically if needed.
    # We will use the color list since we know we have exactly 2 columns.
    try:
        st.line_chart(chart_df, color=["#FFFFFF", "#00E5FF"], height=200)
    except:
        # Fallback if dimensions mismatch momentarily
        st.line_chart(chart_df, height=200)

# EXPORT
csv_buffer = pd.DataFrame(st.session_state.nodes, columns=['X', 'Y']).to_csv(index=False).encode('utf-8')
st.download_button("💾 EXPORT CSV", data=csv_buffer, file_name="topology.csv", mime="text/csv")

# AUTO-LOOP
if is_running:
    time.sleep(0.01)
    st.rerun()
