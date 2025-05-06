# GraphSense: Network Analysis and Visualization Tool

GraphSense is an open-source desktop application designed for analyzing and visualizing complex networks. Built with Python, PyQt5, NetworkX, and Matplotlib, it provides an intuitive interface for loading graph data, performing advanced network analysis, and generating interactive visualizations.

## Features
- **Graph Loading**: Import nodes and edges from CSV files, supporting both directed and undirected graphs.
- **Visualization**: Choose from multiple layout algorithms (e.g., spring, circular, tree, radial) with customizable node/edge colors, sizes, and shapes.
- **Community Detection**: Apply Louvain and Girvan-Newman algorithms to identify community structures, with metrics like modularity, conductance, and silhouette scores.
- **Centrality Measures**: Calculate Degree, Betweenness, Closeness, Eigenvector, and PageRank centralities for node importance analysis.
- **Filtering**: Dynamically filter nodes based on centrality measures or community membership.
- **Interactivity**: Drag nodes, zoom, and pan for an interactive visualization experience.
- **Resource Monitoring**: Ensures efficient performance with CPU and memory checks during heavy computations.

## Installation
### Prerequisites
- Python 3.8+
- pip for installing dependencies

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/AbdoTarek2211/GraphSense.git
   cd netviz
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Deploy the application:
   ```bash
   pyinstaller --onefile --windowed --hidden-import=matplotlib --hidden-import=PyQt5 --hidden-import=psutil sna_desktop_app_v2.py
   ```
4. Run the application in the dist folder

### Requirements
See `requirements.txt` for a complete list. Key dependencies include:
- `networkx`
- `matplotlib`
- `pandas`
- `numpy`
- `PyQt5`
- `python-louvain`
- `psutil`
- `scikit-learn`
- `pyinstaller`

## Usage
1. **Load Data**:
   - Click "Load Nodes CSV" and "Load Edges CSV" to import your graph data.
   - Nodes CSV must include an `ID` column; edges CSV must include `source` and `target` columns.
   - Optional: Include additional attributes (e.g., weight, labels) for enhanced analysis.
2. **Configure Graph**:
   - Select directed/undirected graph type.
   - Choose layout, node size/color attributes, and community detection algorithms.
3. **Analyze**:
   - Click "Calculate Metrics" for network statistics (e.g., density, clustering).
   - Use "Detect Communities" for community analysis.
   - Run "Link Analysis" for centrality measures.
4. **Visualize**:
   - Interact with the graph in the Visualization tab (drag nodes, zoom, pan).
   - Apply filters to focus on specific nodes or communities.

## Example Data
Sample datasets are available in the `data/` folder:
- `nodes.csv`: Node IDs and attributes
- `edges.csv`: Source, target, and optional weights

## Acknowledgments
- Built with [NetworkX](https://networkx.org/) for graph analysis.
- Visualization powered by [Matplotlib](https://matplotlib.org/) and [PyQt5](https://www.riverbankcomputing.com/software/pyqt/).
- Community detection using [python-louvain](https://github.com/taynaud/python-louvain).

## Contact
For questions or feedback, reach out via [GitHub Issues](https://github.com/AbdoTarek2211/GraphSense/issues) or connect with me on [LinkedIn](https://www.linkedin.com/in/abdelrahman-tarek-m).

Happy analyzing! 🚀
