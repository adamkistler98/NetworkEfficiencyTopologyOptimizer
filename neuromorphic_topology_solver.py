import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import io
import pandas as pd
import time

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Neuromorphic Topology Engine", 
    layout="wide", 
    page_icon="🕸️",
    initial_sidebar_state="expanded"
)

# Cyberpunk/Sci-Fi UI Theme
st.markdown("""
<style>
    /* Global Reset */
    .stApp { background-color: #050505; color: #E0E0E0; }
    
    /* HUD Panels */
    div[data-testid="stMetric"] {
        background-color: #111;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 4px;
        border-left: 3px solid #00FF41;
    }
    
    /* Headers */
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier New', monospace; letter-spacing: -1px; }
    
    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #222;
    }
    
    /* Buttons */
    .stButton>button {
        color: #00FF41;
        border: 1px solid #00FF41;
        background: transparent;
        font-family: 'Courier New', monospace;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #00FF41;
        color: #000;
        box-shadow: 0 0 10px #00FF41;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MATH & PHYSICS KERNEL ---

def calculate_mst_cost(nodes):
    """Calculates the Euclidean cost of the Minimum Spanning Tree for comparison."""
    if len(nodes) < 2: return 0.0
    dist_mat = distance_matrix(nodes, nodes)
    mst = minimum_spanning_tree(dist_mat)
    return mst.toarray().sum()

class PhysarumEngine:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        
        # Agents: [x, y, angle]
        self.agents = np.zeros((num_agents, 3))
        # Center initialization (looks cooler than random scatter)
        self.agents[:, 0] = np.random.uniform(width*0.4, width*0.6, num_agents)
        self.agents[:, 1] = np.random.uniform(height*0.4, height*0.6, num_agents)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, sensor_angle, sensor_dist, turn_speed, speed, decay, nodes):
        self.steps += 1
        
        # 1. Sensing (Vectorized)
        angles = self.agents[:, 2]
        # Sensor positions
        l_angle = angles - sensor_angle
        r_angle = angles + sensor_angle
        
        # Wrap coordinates
        def get_pos(a):
            x = (self.agents[:, 0] + np.cos(a) * sensor_dist) % self.width
            y = (self.agents[:, 1] + np.sin(a) * sensor_dist) % self.height
            return x.astype(int), y.astype(int)

        lx, ly = get_pos(l_angle)
        cx, cy = get_pos(angles)
        rx, ry = get_pos(r_angle)
        
        # Sample map
        l_val = self.trail_map[ly, lx]
        c_val = self.trail_map[cy, cx]
        r_val = self.trail_map[ry, rx]
        
        # 2. Steering Logic (Physarum Polycephalum behavior)
        # Random jitter to prevent getting stuck in loops
        jitter = np.random.uniform(-0.1, 0.1, self.num_agents)
        
        move_fwd = (c_val > l_val) & (c_val > r_val)
        move_left = (l_val > c_val) & (l_val > r_val)
        move_right = (r_val > c_val) & (r_val > l_val)
        
        # Apply rotation
        new_angles = angles.copy()
        new_angles[move_left] -= turn_speed
        new_angles[move_right] += turn_speed
        # If ambiguous, rotate randomly
        mask_random = ~(move_fwd | move_left | move_right)
        new_angles[mask_random] += jitter[mask_random] * 5 
        
        self.agents[:, 2] = new_angles

        # 3. Movement
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        # 4. Deposition
        ix = self.agents[:, 0].astype(int)
        iy = self.agents[:, 1].astype(int)
        # Fast deposit using bin count logic (add.at)
        np.add.at(self.trail_map, (iy, ix), 1.0) 
        
        # 5. Node Gravity (Reinforce data centers)
        # This simulates "food" sources keeping the mold alive
        for sx, sy in nodes:
             # Add a Gaussian blob of "food" at each node
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 2.0

        # 6. Global Decay & Diffusion
        # Diffusion spreads the trail, decay removes unused paths
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 3. SESSION MANAGEMENT ---

if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'nodes' not in st.session_state:
    # Default: A random scatter
    st.session_state.nodes = np.random.randint(20, 280, size=(6, 2)).tolist()
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Bio-Luminescence"

# --- 4. SIDEBAR CONTROLS ---

st.sidebar.markdown("## 🎛️ SYSTEM CONTROLS")

# Simulation State
is_running = st.sidebar.toggle("🔴 ONLINE / OFFLINE", value=True)

# Topology Presets
st.sidebar.markdown("### 🌐 Topology Scenarios")
preset = st.sidebar.selectbox("Load Preset:", ["Random Scatter", "Ring Network", "Grid Matrix", "Star Hub"])

if st.sidebar.button("⚠️ LOAD SCENARIO"):
    st.session_state.sim = None # Reset engine
    if preset == "Random Scatter":
        st.session_state.nodes = np.random.randint(20, 280, size=(7, 2)).tolist()
    elif preset == "Ring Network":
        center = (150, 150)
        radius = 100
        angles = np.linspace(0, 2*np.pi, 9)[:-1]
        st.session_state.nodes = [[center[0] + radius*np.cos(a), center[1] + radius*np.sin(a)] for a in angles]
    elif preset == "Grid Matrix":
        st.session_state.nodes = [[x, y] for x in range(60, 260, 60) for y in range(60, 260, 60)]
    elif preset == "Star Hub":
        nodes = [[150, 150]] # Center
        radius = 120
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        nodes.extend([[150 + radius*np.cos(a), 150 + radius*np.sin(a)] for a in angles])
        st.session_state.nodes = nodes
    st.rerun()

# Parameters
with st.sidebar.expander("🧬 Bio-Parameters (Advanced)", expanded=False):
    agent_count = st.slider("Packet Load", 1000, 20000, 8000, step=1000)
    decay_rate = st.slider("Path Decay", 0.85, 0.99, 0.94, step=0.01)
    sensor_angle = st.slider("Sensor Angle", 0.1, 1.5, 0.8)
    speed = st.slider("Propagation Speed", 0.5, 5.0, 2.0)

# View Options
st.session_state.view_mode = st.sidebar.radio("Display Mode", ["Bio-Luminescence", "Skeleton (Routing)", "Thermal"])

# --- 5. INITIALIZATION ---

if st.session_state.sim is None or st.session_state.sim.num_agents != agent_count:
    st.session_state.sim = PhysarumEngine(300, 300, agent_count)

engine = st.session_state.sim
nodes_arr = np.array(st.session_state.nodes)

# --- 6. SIMULATION LOOP & RENDER ---

# Run physics steps if online
if is_running:
    steps_per_frame = 15
    for _ in range(steps_per_frame):
        engine.step(sensor_angle, 9, 0.4, speed, decay_rate, st.session_state.nodes)

# METRICS CALCULATION
mst_cost = calculate_mst_cost(nodes_arr)
# Estimate biological cost (sum of high-traffic pixels)
bio_mask = engine.trail_map > 1.0
bio_cost = np.sum(bio_mask) / 10.0 # Arbitrary scaling for UI comparison

# DASHBOARD HEADER
c1, c2, c3, c4 = st.columns(4)
c1.metric("EPOCH", f"{engine.steps}", delta=None)
c2.metric("NODES", f"{len(st.session_state.nodes)}")
c3.metric("MST COST (Optimal)", f"{int(mst_cost)}")
c4.metric("BIO COST (Actual)", f"{int(bio_cost)}", delta=f"{int(bio_cost - mst_cost)}")

# VISUALIZATION
col_main, col_data = st.columns([3, 1])

with col_main:
    # Prepare Image
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#050505')
    
    # Render Logic
    if st.session_state.view_mode == "Skeleton (Routing)":
        # Threshold logic to show "hard lines"
        disp_map = np.where(engine.trail_map > 1.0, 1.0, 0.0)
        cmap = 'binary_r'
        ax.set_title("Network Topology Extraction", color='white')
    elif st.session_state.view_mode == "Thermal":
        disp_map = engine.trail_map
        cmap = 'inferno'
        ax.set_title("Traffic Density Heatmap", color='white')
    else: # Bio-Luminescence
        # Logarithmic scale for glow effect
        disp_map = np.log1p(engine.trail_map)
        cmap = 'gist_ncar' # Good "tech" look
        ax.set_title("Active Packet Propagation", color='white')

    ax.imshow(disp_map, cmap=cmap, origin='upper', aspect='equal')
    
    # Overlay Nodes
    ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='white', s=150, edgecolors='#00FF41', linewidth=2, zorder=10)
    
    # Draw MST Lines (Ghost lines for comparison)
    if st.toggle("Show Optimal Reference (MST)", value=False):
        dm = distance_matrix(nodes_arr, nodes_arr)
        mst = minimum_spanning_tree(dm).toarray()
        for i in range(len(nodes_arr)):
            for j in range(len(nodes_arr)):
                if mst[i, j] > 0:
                    p1 = nodes_arr[i]
                    p2 = nodes_arr[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='white', alpha=0.3, linestyle='--', linewidth=1)

    ax.axis('off')
    st.pyplot(fig, use_container_width=True)

# DATA PANEL
with col_data:
    st.markdown("### 📊 Live Telemetry")
    
    # Efficiency calculation
    if mst_cost > 0:
        eff_ratio = (mst_cost / (bio_cost + 1)) * 100
        # Normalize for display (slime is usually "longer" than straight lines)
        efficiency = min(100, eff_ratio * 4.0) 
        st.progress(int(efficiency)/100, text=f"Topology Efficiency: {int(efficiency)}%")
    
    st.markdown("---")
    st.markdown("**Network Nodes**")
    df = pd.DataFrame(st.session_state.nodes, columns=["Lat", "Long"])
    st.dataframe(df, height=150, hide_index=True)
    
    st.markdown("---")
    st.markdown("**Download Reports**")
    
    # Generate CSV
    csv_buf = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Export Node Map", data=csv_buf, file_name="topology.csv", mime="text/csv", use_container_width=True)
    
    # Generate Snapshot
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', facecolor='#050505', bbox_inches='tight')
    st.download_button("📸 Capture Topology", data=img_buf.getvalue(), file_name="network_state.png", mime="image/png", use_container_width=True)

# AUTO-LOOP
if is_running:
    time.sleep(0.01)
    st.rerun()
