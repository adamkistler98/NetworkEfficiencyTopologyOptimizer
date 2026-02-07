import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import io

# --- 1. DARK MODE & UI CONFIG ---
st.set_page_config(
    page_title="Neuromorphic Topology V6", 
    layout="wide", 
    page_icon="🕸️",
    initial_sidebar_state="expanded"
)

# HARDCODED STEALTH CSS (Fixes white backgrounds)
st.markdown("""
<style>
    /* MAIN CONTAINER - FORCE BLACK */
    .stApp { background-color: #000000; color: #00E5FF; }
    
    /* REMOVE WHITE BACKGROUNDS FROM PLOTS */
    div[data-testid="stImage"] { background-color: transparent !important; }
    
    /* INPUTS & DROPDOWNS */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #0a0a0a !important;
        border: 1px solid #333 !important;
        color: #00E5FF !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover { border-color: #00E5FF !important; }
    
    /* BUTTONS */
    .stButton>button {
        color: #00E5FF !important;
        border: 1px solid #333 !important;
        background-color: #050505 !important;
    }
    .stButton>button:hover {
        border-color: #00E5FF !important;
        box-shadow: 0 0 10px rgba(0,229,255,0.2);
    }

    /* TEXT & METRICS */
    h1, h2, h3, h4, p, label { color: #00E5FF !important; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 24px; }
    div[data-testid="stMetricLabel"] { color: #888888 !important; }
    
    /* HIDE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS ENGINE (The Slime) ---

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

    def step(self, nodes, speed=2.0, decay=0.95):
        self.steps += 1
        # 1. SENSING
        sensor_angle = 0.7
        sensor_dist = 9.0
        angles = self.agents[:, 2]
        
        # Helper for toroidal wrapping
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
        
        # 2. DECISION (Steer towards highest pheromone)
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
        
        # 5. NODE ATTRACTION (Food)
        for sx, sy in nodes:
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 5.0 # Strong gravity

        # 6. DECAY
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 3. SESSION STATE ---
if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = [[100, 100], [200, 100], [200, 200], [100, 200], [150, 150]] # Default grid
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. SIDEBAR ---
st.sidebar.title("🎛️ SYSTEM KERNEL")
is_running = st.sidebar.toggle("🟢 SYSTEM ONLINE", value=True)

st.sidebar.markdown("### 🗺️ TOPOLOGY PRESETS")
preset = st.sidebar.selectbox("Select Layout", ["Square Loop", "Pentagon", "Random Scatter", "Hub & Spoke"])
if st.sidebar.button("⚠️ LOAD PRESET"):
    st.session_state.sim = None
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

st.sidebar.markdown("### ⚡ RESILIENCE TEST")
if st.sidebar.button("💥 DESTROY RANDOM NODE"):
    if len(st.session_state.nodes) > 2:
        st.session_state.nodes.pop(np.random.randint(0, len(st.session_state.nodes)))
        # Do not reset sim, let it adapt!

# --- 5. INITIALIZE ---
if st.session_state.sim is None:
    st.session_state.sim = BioEngine(300, 300, 4000)

engine = st.session_state.sim
nodes_arr = np.array(st.session_state.nodes)

# --- 6. MAIN LOOP ---
if is_running:
    for _ in range(15): # Fast forward
        engine.step(st.session_state.nodes)

# --- 7. UI LAYOUT ---
st.title("NEUROMORPHIC TOPOLOGY SOLVER v6")

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
c1.metric("NODES ONLINE", f"{len(nodes_arr)}")
c2.metric("EPOCH", f"{engine.steps}")
# Calculate MST
if len(nodes_arr) > 1:
    dist_mat = distance_matrix(nodes_arr, nodes_arr)
    mst_cost = minimum_spanning_tree(dist_mat).toarray().sum()
    c3.metric("MST COST (OPTIMAL)", f"{int(mst_cost)}")
else:
    c3.metric("MST COST", "0")
    
# Bio Cost (Sum of trail density)
bio_cost = np.sum(engine.trail_map > 1.0) / 10
c4.metric("BIO-SOLVER COST", f"{int(bio_cost)}")

st.markdown("---")

# --- 8. DUAL GRAPHICS (SIDE BY SIDE) ---
col_left, col_right = st.columns(2)

# LEFT: THE ORGANIC PROCESS
with col_left:
    st.markdown("#### 🧬 BIO-PROCESS (Solving...)")
    fig1, ax1 = plt.subplots(figsize=(5, 5), facecolor='black')
    
    # Render Slime
    disp_map = np.log1p(engine.trail_map)
    ax1.imshow(disp_map, cmap='magma', origin='upper', vmin=0, vmax=np.percentile(disp_map, 99))
    
    # Render Nodes
    if len(nodes_arr) > 0:
        ax1.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='white', s=50, edgecolors='cyan')
    
    ax1.axis('off')
    fig1.tight_layout(pad=0)
    st.pyplot(fig1, use_container_width=True)

# RIGHT: THE EXTRACTED SOLUTION
with col_right:
    st.markdown("#### 💠 DIGITAL TWIN (Solution)")
    fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor='black')
    
    # 1. Dark Background
    ax2.set_facecolor('black')
    ax2.set_xlim(0, 300)
    ax2.set_ylim(300, 0) # Invert Y to match image
    
    # 2. Extract Lines (Thresholding)
    # This simulates "vectorizing" the slime trail
    y_trail, x_trail = np.where(engine.trail_map > 2.0) # High traffic areas
    if len(x_trail) > 0:
        # Plot as a scatter of small points to simulate connections
        # (Faster than actual contour fitting for real-time)
        ax2.scatter(x_trail, y_trail, c='#00E5FF', s=1, alpha=0.3, label='Active Route')
    
    # 3. Draw Optimal MST Lines (Ghost lines for comparison)
    if len(nodes_arr) > 1:
        mst_matrix = minimum_spanning_tree(distance_matrix(nodes_arr, nodes_arr)).toarray()
        for i in range(len(nodes_arr)):
            for j in range(len(nodes_arr)):
                if mst_matrix[i, j] > 0:
                    p1, p2 = nodes_arr[i], nodes_arr[j]
                    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], c='white', alpha=0.3, linestyle='--', linewidth=0.5)

    # 4. Render Nodes
    if len(nodes_arr) > 0:
        ax2.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='#00E5FF', s=100, marker='s', edgecolors='white')
        for i, (nx, ny) in enumerate(nodes_arr):
            ax2.text(nx+5, ny-5, f"N{i}", color='white', fontsize=8)

    ax2.axis('off')
    fig2.tight_layout(pad=0)
    st.pyplot(fig2, use_container_width=True)

# --- 9. TELEMETRY & EXPORT ---
st.markdown("#### 📉 CONVERGENCE HISTORY")
st.session_state.history.append({"MST": mst_cost if len(nodes_arr)>1 else 0, "BIO": bio_cost})
if len(st.session_state.history) > 100: st.session_state.history.pop(0)

chart_data = pd.DataFrame(st.session_state.history)
st.line_chart(chart_data, color=["#FFFFFF", "#00E5FF"], height=150)

# EXPORT
csv_buffer = pd.DataFrame(st.session_state.nodes, columns=['X', 'Y']).to_csv().encode('utf-8')
st.download_button("💾 DOWNLOAD NODE TOPOLOGY (CSV)", data=csv_buffer, file_name="topology.csv", mime="text/csv")

# AUTO-LOOP
if is_running:
    time.sleep(0.01)
    st.rerun()
