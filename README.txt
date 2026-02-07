# Neuromorphic Topology Solver (NTS)
### Bio-Mimetic Network Optimization Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scientific Stack](https://img.shields.io/badge/SciPy%20%7C%20NumPy-Optimized-green)

## 📡 Overview
The **Neuromorphic Topology Solver** is a Bio-Mimetic simulation engine designed to solve **Steiner Tree** optimization problems for distributed network infrastructures. 

By replicating the biological intelligence of *Physarum polycephalum* (Slime Mold), this tool autonomously discovers the most efficient routing paths between server nodes. It utilizes a **Vectorized Agent Model** to simulate thousands of independent "packets" that exhibit emergent behavior, naturally converging on a network topology that balances minimal latency with maximum redundancy.

## 🛠️ Key Capabilities
* **Autonomous Routing:** Agents utilize chemotaxis-based heuristics to discover data centers without pre-programmed paths.
* **Vectorized Physics Engine:** Built on NumPy to handle 5,000+ concurrent agents with real-time frame rates.
* **Dynamic Optimization:** Users can adjust "Decay Rates" (Pruning Pressure) and "Sensor Angles" to simulate different network conditions (Mesh vs. Backbone).
* **Enterprise Reporting:** Includes built-in tools to snapshot high-resolution topology maps and export node telemetry as CSV data.

## 🧠 The Algorithm (Bio-Computing)
This solver rejects standard Dijkstra or A* algorithms in favor of a stochastic, self-healing approach:
1.  **Exploration:** Agents move stochastically through the void, depositing a digital pheromone trail.
2.  **Sensation:** Agents scan a 3-point wide-angle horizon. If they detect a trail or a node, they steer toward it.
3.  **Reinforcement:** Successful paths are traversed more frequently, increasing their signal strength (Bandwidth).
4.  **Decay:** The environment applies a Gaussian decay filter. Inefficient or redundant paths "evaporate," leaving only the optimal trunk lines.

## 🚀 Installation & Usage

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/neuromorphic-topology-solver.git](https://github.com/your-username/neuromorphic-topology-solver.git)
cd neuromorphic-topology-solver

├── neuromorphic_topology_solver.py  # Main Application Core
├── requirements.txt                 # Dependency Manifest
├── README.md                        # Documentation
└── /assets                          # Screenshots and Architecture Diagrams

📊 Telemetry & Export
The application provides a "Net-Ops Console" allowing for:

Live Metrics: Tracking active cabling cost (km) and optimization epochs.

Snapshotting: High-res PNG downloads of the current convergence state.

Data Export: CSV serialization of node coordinates and efficiency metrics for external analysis.

📜 License
Distributed under the MIT License. See LICENSE for more information.

Engineered by Adam Kistler | Cybersecurity & Platform Administration
