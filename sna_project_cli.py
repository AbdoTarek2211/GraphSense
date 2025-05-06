import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import ipywidgets as widgets
import time
import warnings
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
from IPython.display import clear_output
from tqdm.notebook import tqdm
from collections import defaultdict
from community import community_louvain
from sklearn.metrics import normalized_mutual_info_score
from itertools import combinations
from networkx.algorithms import community as nx_community

warnings.filterwarnings('ignore')

class EnhancedNetworkAnalysisTool:
    def __init__(self):
        self.G = None
        self.directed = False
        self.node_attributes = {}
        self.edge_attributes = {}
        self.current_layout = 'spring'
        self.community_results = {}
        self.centrality_measures = {}
        self.filtered_graph = None
        self.centrality_ranges = {}  # Store ranges for centrality measures

        # Initialize widgets
        self.init_widgets()

    def init_widgets(self):
        # Graph type
        self.directed_checkbox = widgets.Checkbox(
            value=False,
            description='Directed Graph',
            disabled=False
        )

        # File upload
        self.nodes_upload = widgets.FileUpload(
            description='Upload Nodes CSV',
            multiple=False
        )

        self.edges_upload = widgets.FileUpload(
            description='Upload Edges CSV',
            multiple=False
        )

        # Color pickers
        self.node_color_picker = widgets.ColorPicker(
            concise=False,
            description='Node color:',
            value='skyblue',
            disabled=False
        )

        self.edge_color_picker = widgets.ColorPicker(
            concise=False,
            description='Edge color:',
            value='gray',
            disabled=False
        )

        # Layout options
        self.layout_dropdown = widgets.Dropdown(
            options=['spring', 'circular', 'random', 'shell', 'spectral', 'kamada_kawai', 'tree', 'radial', 'stress', 'multipartite'],
            value='spring',
            description='Layout:',
            disabled=False
        )

        # Node styling
        self.node_size_dropdown = widgets.Dropdown(
            options=['uniform'],
            value='uniform',
            description='Node size by:',
            disabled=False
        )

        self.node_color_dropdown = widgets.Dropdown(
            options=['uniform'],
            value='uniform',
            description='Node color by:',
            disabled=False
        )

        self.node_shape_dropdown = widgets.Dropdown(
            options=['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', '8'],
            value='o',
            description='Node shape:',
            disabled=False
        )

        self.show_labels_checkbox = widgets.Checkbox(
            value=False,
            description='Show labels',
            disabled=False
        )

        # Filtering
        self.filter_dropdown = widgets.Dropdown(
            options=["None", "Degree Centrality", "Betweenness Centrality",
                     "Closeness Centrality", "Eigenvector Centrality",
                     "PageRank"],
            value="None",
            description='Filter by:',
            disabled=False
        )

        self.min_slider = widgets.FloatSlider(
            value=0,
            min=0,
            max=1,
            step=0.01,
            description='Min threshold:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.4f'
        )

        self.max_slider = widgets.FloatSlider(
            value=1,
            min=0,
            max=1,
            step=0.01,
            description='Max threshold:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.4f'
        )

        # Reset button for sliders
        self.reset_button = widgets.Button(
            description='Reset Sliders',
            button_style='warning',
            tooltip='Reset sliders if they get into a bad state',
            layout=widgets.Layout(display='none')  # Initially hidden
        )
        self.reset_button.on_click(lambda b: self.reset_sliders())

        # Community detection
        self.community_dropdown = widgets.Dropdown(
            options=["Louvain", "Girvan-Newman", "All"],
            value="Louvain",
            description='Community Detection Algorithm:',
            disabled=False
        )

        self.reset_filter_button = widgets.Button(
            description="Reset Filters",
            button_style='warning'
        )

        # Buttons
        self.load_button = widgets.Button(description="Load Graph")
        self.load_button.on_click(self.load_graph)

        self.metrics_button = widgets.Button(description="Calculate Metrics")
        self.metrics_button.on_click(self.show_metrics)

        self.community_button = widgets.Button(description="Detect Communities")
        self.community_button.on_click(self.detect_communities)

        self.link_button = widgets.Button(description="Link Analysis")
        self.link_button.on_click(self.link_analysis)

        self.apply_filter_button = widgets.Button(description="Apply Filter")
        self.apply_filter_button.on_click(self.apply_filter)
        self.reset_filter_button.on_click(self.reset_filters)

        # Output areas
        self.graph_output = widgets.Output()
        self.metrics_output = widgets.Output()
        self.community_output = widgets.Output()
        self.link_output = widgets.Output()

        # Display all widgets
        display(widgets.VBox([
            widgets.HBox([self.directed_checkbox, self.nodes_upload, self.edges_upload]),
            widgets.HBox([self.node_color_picker, self.edge_color_picker]),
            widgets.HBox([self.layout_dropdown, self.node_size_dropdown, self.node_color_dropdown]),
            widgets.HBox([self.node_shape_dropdown, self.show_labels_checkbox]),
            widgets.HBox([self.filter_dropdown, self.community_dropdown, self.min_slider, self.max_slider]),
            widgets.HBox([self.reset_filter_button, self.reset_button]),
            widgets.HBox([self.load_button, self.metrics_button, self.community_button,
                         self.link_button, self.apply_filter_button]),
            widgets.Tab([
                self.graph_output,
                self.metrics_output,
                self.community_output,
                self.link_output
            ], titles=['Graph', 'Metrics', 'Communities', 'Link Analysis'])
        ]))
        self.filter_dropdown.observe(self.on_filter_change, names='value')

    def on_filter_change(self, change):
        """Handle filter dropdown changes with error recovery"""
        try:
            self.update_filter_options(change)
            self.reset_button.layout.display = 'none'
        except Exception as e:
            print(f"Error updating filter: {e}")
            self.reset_button.layout.display = 'block'

    def _safe_update_single_slider(self, slider, new_min, new_max, new_value):
        """Safely update a single slider's attributes"""
        try:
            if new_min >= new_max:
                print(f"Warning: Invalid slider range ({new_min} >= {new_max}), applying fix")
                new_max = new_min + max(new_min * 0.1, 0.001)

            safe_value = max(new_min, min(new_value, new_max))

            with slider.hold_sync():
                slider.value = (slider.min + slider.max) / 2
                slider.min = new_min
                slider.max = new_max
                slider.value = safe_value

            print(f"Slider updated: range [{new_min:.6f}, {new_max:.6f}], value {safe_value:.6f}")
            return True
        except Exception as e:
            print(f"Error in _safe_update_single_slider: {str(e)}")
            self.reset_button.layout.display = 'block'
            return False

    def reset_sliders(self, silent=False):
        """Reset sliders to default state"""
        try:
            with self.min_slider.hold_sync(), self.max_slider.hold_sync():
                self.min_slider.value = 0.5
                self.max_slider.value = 0.5
                self.min_slider.min = 0
                self.min_slider.max = 1
                self.max_slider.min = 0
                self.max_slider.max = 1
                self.min_slider.description = 'Min threshold:'
                self.max_slider.description = 'Max threshold:'
                self.min_slider.readout_format = '.4f'
                self.max_slider.readout_format = '.4f'
                self.min_slider.step = 0.01
                self.max_slider.step = 0.01
                self.min_slider.value = 0
                self.max_slider.value = 1

            if not silent:
                print("Sliders reset successfully")
            self.reset_button.layout.display = 'none'
        except Exception as e:
            print(f"Error resetting sliders: {str(e)}")
            self.reset_button.layout.display = 'block'

    def update_filter_options(self, change):
        """Update slider ranges and options based on the selected filter"""
        filter_type = change.new
        self.reset_sliders(silent=True)

        with self.min_slider.hold_sync(), self.max_slider.hold_sync():
            self.min_slider.description = 'Min threshold:'
            self.max_slider.description = 'Max threshold:'
            self.min_slider.readout_format = '.4f'
            self.max_slider.readout_format = '.4f'
            self.min_slider.step = 0.01
            self.max_slider.step = 0.01

        if filter_type == "None":
            self._safe_update_single_slider(self.min_slider, 0, 1, 0)
            self._safe_update_single_slider(self.max_slider, 0, 1, 1)
            return

        filter_to_centrality = {
            "Degree Centrality": "Degree",
            "Betweenness Centrality": "Betweenness",
            "Closeness Centrality": "Closeness",
            "Eigenvector Centrality": "Eigenvector",
            "PageRank": "PageRank"
        }

        centrality_name = filter_to_centrality.get(filter_type)
        if centrality_name in self.centrality_measures:
            if centrality_name not in self.centrality_ranges:
                print(f"Please run Link Analysis to calculate {filter_type} first")
                return

            min_val = self.centrality_ranges[centrality_name]['min']
            max_val = self.centrality_ranges[centrality_name]['max']

            if filter_type == "PageRank":
                readout_format = '.6f'
                step = 0.00001
            elif filter_type == "Betweenness Centrality":
                readout_format = '.6f'
                step = 0.00001
            elif filter_type == "Closeness Centrality":
                readout_format = '.6f'
                step = 0.0001
            else:
                readout_format = '.4f'
                step = 0.001

            if abs(max_val - min_val) < 1e-10:
                min_val = max(0, min_val * 0.9)
                max_val = min_val * 1.1

            buffer = max((max_val - min_val) * 0.01, 1e-10)
            min_val = max(0, min_val - buffer)
            max_val = max_val + buffer

            print(f"Setting {filter_type} range: {min_val:.6f} to {max_val:.6f}")

            with self.min_slider.hold_sync(), self.max_slider.hold_sync():
                self.min_slider.description = 'Min value:'
                self.max_slider.description = 'Max value:'
                self.min_slider.readout_format = readout_format
                self.max_slider.readout_format = readout_format
                self.min_slider.step = step
                self.max_slider.step = step

            self._safe_update_single_slider(self.min_slider, min_val, max_val, min_val)
            self._safe_update_single_slider(self.max_slider, min_val, max_val, max_val)
        else:
            print(f"Please calculate {filter_type} first using Link Analysis")

    def reset_filters(self, b):
        """Reset all filters to default state"""
        self.filter_dropdown.value = "None"
        self.min_slider.value = self.min_slider.min
        self.max_slider.value = self.max_slider.max
        self.filtered_graph = None

        with self.graph_output:
            print("Filters reset. Showing full graph.")
        self.draw_graph()

    def apply_filter(self, b):
        """Apply selected filters to the graph"""
        if self.G is None:
            print("Please load a graph first")
            return

        filter_type = self.filter_dropdown.value
        min_val = self.min_slider.value
        max_val = self.max_slider.value
        self.filtered_graph = None

        if filter_type == "None":
            self.draw_graph()
            return

        filter_to_centrality = {
            "Degree Centrality": "Degree",
            "Betweenness Centrality": "Betweenness",
            "Closeness Centrality": "Closeness",
            "Eigenvector Centrality": "Eigenvector",
            "PageRank": "PageRank"
        }

        centrality_name = filter_to_centrality.get(filter_type)
        if centrality_name not in self.centrality_measures:
            print(f"Please calculate {filter_type} first using Link Analysis")
            return

        centrality = self.centrality_measures[centrality_name]
        nodes_to_keep = [node for node, score in centrality.items() if min_val <= score <= max_val]

        if not nodes_to_keep:
            with self.graph_output:
                clear_output(wait=True)
                print("Filter removed all nodes. Try adjusting the threshold values.")
                self.filtered_graph = self.G.subgraph([])
                self.draw_graph(self.filtered_graph)
            return

        self.filtered_graph = self.G.subgraph(nodes_to_keep)

        with self.graph_output:
            print(f"Showing filtered graph with {self.filtered_graph.number_of_nodes()} nodes and "
                  f"{self.filtered_graph.number_of_edges()} edges")
        self.draw_graph(self.filtered_graph)

    def load_graph(self, b):
        if not self.nodes_upload.value or not self.edges_upload.value:
            print("Please upload both nodes and edges files")
            return

        try:
            nodes_content = next(iter(self.nodes_upload.value.values()))['content']
            self.node_df = pd.read_csv(io.BytesIO(nodes_content))

            if 'ID' not in self.node_df.columns:
                raise ValueError("Nodes CSV must contain an 'ID' column")

            self.node_attributes = self.node_df.set_index('ID').to_dict('index')
            attributes = [col for col in self.node_df.columns if col != 'ID']
            self.node_size_dropdown.options = ['uniform'] + attributes
            self.node_color_dropdown.options = ['uniform'] + attributes

            edges_content = next(iter(self.edges_upload.value.values()))['content']
            self.edge_df = pd.read_csv(io.BytesIO(edges_content))
            self.edge_df.columns = self.edge_df.columns.str.lower()

            if not {'source', 'target'}.issubset(self.edge_df.columns):
                raise ValueError("Edges CSV must contain 'source' and 'target' columns")

            if self.edge_df.duplicated(subset=['source', 'target']).any():
                print("Warning: Duplicate edges found. Aggregating them...")
                if 'weight' in self.edge_df.columns:
                    self.edge_df = self.edge_df.groupby(['source', 'target'])['weight'].sum().reset_index()
                else:
                    self.edge_df = self.edge_df.drop_duplicates(subset=['source', 'target'])

            self.edge_attributes = self.edge_df.to_dict('records')
            self.directed = self.directed_checkbox.value
            self.G = nx.DiGraph() if self.directed else nx.Graph()

            for node, attrs in self.node_attributes.items():
                self.G.add_node(node, **attrs)

            for edge in self.edge_attributes:
                source = edge['source']
                target = edge['target']
                edge_attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
                self.G.add_edge(source, target, **edge_attrs)

            self.filtered_graph = None
            self.draw_graph()

        except Exception as e:
            print(f"Error loading files: {str(e)}")

    def draw_graph(self, graph=None):
        if graph is None:
            graph = self.G if self.filtered_graph is None else self.filtered_graph

        if graph is None or graph.number_of_nodes() == 0:
            print("No graph to display or graph is empty")
            return

        recommended_layouts = ['spring', 'multipartite', 'tree'] if self.directed else list(self.layout_dropdown.options)

        with self.graph_output:
            clear_output(wait=True)
            base_width = 12
            if 'community' in graph.nodes[list(graph.nodes())[0]]:
                communities = [graph.nodes[n]['community'] for n in graph.nodes()]
                n_communities = len(set(communities))
                extra_width = min(n_communities // 5, 6)
                plt.figure(figsize=(base_width + extra_width, 10))
            else:
                plt.figure(figsize=(base_width, 10))

            layout = self.layout_dropdown.value
            if self.directed and layout not in recommended_layouts:
                print(f"Warning: {layout} layout may not work well with directed graphs")

            pos = self._compute_layout(graph, layout)
            if pos is None:
                print("Using spring layout as fallback")
                pos = nx.spring_layout(graph, seed=42)

            size_attr = self.node_size_dropdown.value
            if size_attr == 'uniform':
                node_size = 300
            else:
                try:
                    sizes = [float(graph.nodes[n].get(size_attr, 1)) * 100 for n in graph.nodes()]
                    node_size = sizes
                except Exception as e:
                    print(f"Error processing size attribute puneet: {e}")
                    node_size = 300

            color_attr = self.node_color_dropdown.value
            if color_attr == 'uniform':
                node_color = self.node_color_picker.value
            else:
                try:
                    colors = [graph.nodes[n].get(color_attr, 0) for n in graph.nodes()]
                    if any(isinstance(c, str) for c in colors):
                        unique_vals = list(set(colors))
                        color_map = {v: i for i, v in enumerate(unique_vals)}
                        colors = [color_map[c] for c in colors]
                    node_color = colors
                except Exception as e:
                    print(f"Error processing color attribute: {e}")
                    node_color = self.node_color_picker.value

            if (color_attr == 'uniform' and list(graph.nodes()) and
                'community' in graph.nodes[list(graph.nodes())[0]]):
                communities = [graph.nodes[n].get('community', 0) for n in graph.nodes()]
                node_color = communities

            edge_widths = []
            for u, v in graph.edges():
                try:
                    weight = graph.get_edge_data(u, v, {}).get('weight', 1.0)
                    edge_widths.append(float(weight) * 2)
                except:
                    edge_widths.append(1.0)

            nx.draw_networkx_edges(graph, pos, alpha=0.5,
                                  edge_color=self.edge_color_picker.value,
                                  width=edge_widths)

            nx.draw_networkx_nodes(graph, pos,
                                   node_size=node_size,
                                   node_color=node_color,
                                   node_shape=self.node_shape_dropdown.value,
                                   cmap=plt.cm.tab20,
                                   alpha=0.8)

            if self.show_labels_checkbox.value:
                nx.draw_networkx_labels(graph, pos, font_size=8)

            if 'community' in graph.nodes[list(graph.nodes())[0]]:
                communities = [graph.nodes[n]['community'] for n in graph.nodes()]
                unique_communities = sorted(set(communities))
                community_counts = {comm: communities.count(comm) for comm in unique_communities}
                cmap = plt.cm.get_cmap('tab20', len(unique_communities))
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               label=f'Community {comm} ({community_counts[comm]} nodes)',
                               markerfacecolor=cmap(i), markersize=10)
                    for i, comm in enumerate(unique_communities)
                ]
                plt.legend(handles=legend_elements, title='Detected Communities',
                           bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

            plt.axis('off')
            algo = self.community_dropdown.value if hasattr(self, 'community_dropdown') else "Louvain"
            plt.title(f"Network Visualization - {layout.capitalize()} Layout | Communities ({algo})")
            plt.show()

    def _compute_layout(self, graph, layout):
        """Compute layout positions, handling disconnected components"""
        try:
            if layout == 'spring':
                return nx.spring_layout(graph, k=0.3, iterations=50, seed=42)
            elif layout == 'circular':
                return nx.circular_layout(graph)
            elif layout == 'random':
                return nx.random_layout(graph, seed=42)
            elif layout in ['shell', 'spectral', 'kamada_kawai']:
                if self.directed or not nx.is_connected(graph.to_undirected(as_view=True)):
                    pos = {}
                    components = list(nx.connected_components(graph.to_undirected(as_view=True)))
                    offset = 0
                    for component in components:
                        subgraph = graph.subgraph(component)
                        if len(component) > 1 or layout != 'spectral':
                            sub_pos = (
                                nx.shell_layout(subgraph) if layout == 'shell' else
                                nx.spectral_layout(subgraph) if layout == 'spectral' else
                                nx.kamada_kawai_layout(subgraph)
                            )
                            for node, coords in sub_pos.items():
                                pos[node] = coords + np.array([offset, 0])
                        else:
                            pos[list(component)[0]] = np.array([offset, 0])
                        offset += 1.5
                    return pos
                return (
                    nx.shell_layout(graph) if layout == 'shell' else
                    nx.spectral_layout(graph) if layout == 'spectral' else
                    nx.kamada_kawai_layout(graph)
                )
            elif layout == 'tree':
                try:
                    if nx.is_tree(graph):
                        root = max(graph.degree(), key=lambda x: x[1])[0]
                        return nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot', root=root)
                    else:
                        root = max(graph.degree(), key=lambda x: x[1])[0]
                        return self._simple_tree_layout(graph, root)
                except:
                    return None
            elif layout == 'radial':
                return self._radial_layout(graph)
            elif layout == 'stress':
                try:
                    return nx.kamada_kawai_layout(graph)
                except:
                    return None
            elif layout == 'multipartite':
                return self._multipartite_layout(graph)
            return None
        except Exception as e:
            print(f"Error computing {layout} layout: {str(e)}")
            return None

    def _simple_tree_layout(self, G, root):
        pos = {root: np.array([0, 0])}
        visited = {root}
        queue = [(root, 0)]
        level_counts = defaultdict(int)

        while queue:
            node, level = queue.pop(0)
            level_counts[level] += 1
            width = level_counts[level]
            pos[node] = np.array([width, -level])

            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, level + 1))

        nodes_by_level = defaultdict(list)
        for node, (x, y) in pos.items():
            nodes_by_level[y].append(node)

        for level, nodes in nodes_by_level.items():
            width = len(nodes)
            for i, node in enumerate(sorted(nodes, key=lambda n: pos[n][0])):
                pos[node][0] = i - width / 2

        return pos

    def _radial_layout(self, G):
        if not G.nodes():
            return {}

        try:
            groups = {n: G.nodes[n].get('community', G.degree(n) % 5) for n in G.nodes()}
        except:
            return {}

        unique_groups = sorted(set(groups.values()))
        pos = {}
        for i, group in enumerate(unique_groups):
            nodes_in_group = [n for n in G.nodes() if groups[n] == group]
            if not nodes_in_group:
                continue
            radius = 0.1 + 0.8 * (i / max(1, len(unique_groups) - 1))
            theta = np.linspace(0, 2 * np.pi, len(nodes_in_group) + 1, endpoint=False)
            for j, node in enumerate(nodes_in_group):
                pos[node] = np.array([radius * np.cos(theta[j]), radius * np.sin(theta[j])])
        return pos

    def _multipartite_layout(self, G):
        if not G.nodes():
            return {}

        try:
            layers = {n: G.nodes[n].get('layer', G.nodes[n].get('community', G.degree(n) % 5)) for n in G.nodes()}
        except:
            return {}

        unique_layers = sorted(set(layers.values()))
        pos = {}
        for i, layer in enumerate(unique_layers):
            nodes_in_layer = [n for n in G.nodes() if layers[n] == layer]
            if not nodes_in_layer:
                continue
            y = 1 - (i / max(1, len(unique_layers) - 1))
            x_positions = np.linspace(0, 1, len(nodes_in_layer)) if len(nodes_in_layer) > 1 else [0.5]
            for j, node in enumerate(nodes_in_layer):
                pos[node] = np.array([x_positions[j], y])
        return pos

    def show_metrics(self, b):
        if self.G is None:
            print("Please load a graph first")
            return

        with self.metrics_output:
            clear_output(wait=True)
            degrees_dict = dict(self.G.degree())
            in_degrees = dict(self.G.in_degree()) if self.directed else None
            out_degrees = dict(self.G.out_degree()) if self.directed else None
            clustering_coeffs = nx.clustering(self.G)
            avg_clustering = nx.average_clustering(self.G)

            metrics = {
                "Number of Nodes": self.G.number_of_nodes(),
                "Number of Edges": self.G.number_of_edges(),
                "Average Degree": sum(degrees_dict.values()) / self.G.number_of_nodes(),
                "Density": nx.density(self.G),
                "Average Clustering": avg_clustering,
                "Transitivity": nx.transitivity(self.G),
                "Diameter": self._calculate_diameter(self.G),
                "Average Path Length": self._calculate_avg_path_length(self.G)
            }

            print("{:<30} {:<15}".format('Metric', 'Value'))
            print("-" * 45)
            for name, value in metrics.items():
                print("{:<30} {:<15}".format(name, str(value)))

            degrees_list = list(degrees_dict.values())
            print("\n{:<30} {:<15}".format('Degree Distribution', ''))
            print("{:<30} {:<15.2f}".format('Min degree', min(degrees_list)))
            print("{:<30} {:<15.2f}".format('Max degree', max(degrees_list)))
            print("{:<30} {:<15.2f}".format('Median degree', np.median(degrees_list)))
            print("{:<30} {:<15.2f}".format('Degree std.dev', np.std(degrees_list)))

            print("\n{:<10} {:<15} {:<15} {:<15}".format(
                'Node',
                'Degree' if not self.directed else 'Total Degree',
                'In-Degree' if self.directed else '',
                'Out-Degree' if self.directed else ''
            ))
            print("-" * 50)
            for node in self.G.nodes():
                degree = degrees_dict.get(node, 0)
                in_deg = in_degrees.get(node, 0) if self.directed else ""
                out_deg = out_degrees.get(node, 0) if self.directed else ""
                print("{:<10} {:<15} {:<15} {:<15}".format(str(node), str(degree), str(in_deg), str(out_deg)))

            if self.directed:
                in_degrees_list = list(in_degrees.values())
                out_degrees_list = list(out_degrees.values())
                print("\n{:<30} {:<15}".format('In-Degree Distribution', ''))
                print("{:<30} {:<15.2f}".format('Average in-degree', np.mean(in_degrees_list)))
                print("{:<30} {:<15.2f}".format('Min in-degree', min(in_degrees_list)))
                print("{:<30} {:<15.2f}".format('Max in-degree', max(in_degrees_list)))
                print("\n{:<30} {:<15}".format('Out-Degree Distribution', ''))
                print("{:<30} {:<15.2f}".format('Average out-degree', np.mean(out_degrees_list)))
                print("{:<30} {:<15.2f}".format('Min out-degree', min(out_degrees_list)))
                print("{:<30} {:<15.2f}".format('Max out-degree', max(out_degrees_list)))

            if clustering_coeffs:
                coeff_values = list(clustering_coeffs.values())
                num_nodes = len(coeff_values)
                print("\n{:<30} {:<15}".format('Clustering Coefficient', ''))
                print("{:<30} {:<15.2f}".format('Average coefficient', avg_clustering))
                print("{:<30} {:<15.2f}".format('Min coefficient', min(coeff_values)))
                print("{:<30} {:<15.2f}".format('Max coefficient', max(coeff_values)))
                print("{:<30} {:<15.2f}".format('Median coefficient', np.median(coeff_values)))
                print("{:<30} {:<15.2f}".format('Coefficient std.dev', np.std(coeff_values)))
                if num_nodes <= 50:
                    print("\nIndividual Clustering Coefficients:")
                    for node, coeff in list(clustering_coeffs.items())[:50]:
                        print(f"Node {node}: {coeff:.4f}")
                elif num_nodes <= 200:
                    sample_nodes = list(clustering_coeffs.keys())[:5]
                    print("\nSample Clustering Coefficients:")
                    for node in sample_nodes:
                        print(f"Node {node}: {clustering_coeffs[node]:.4f}")
                    print(f"\n... and {num_nodes-5} more nodes")
                else:
                    print("\n(Network too large to display individual coefficients)")

    def _calculate_diameter(self, G):
        """Calculate diameter for directed or undirected graphs, handling disconnected components"""
        if self.directed:
            components = list(nx.strongly_connected_components(G))
            if not components:
                return "Disconnected (no components)"
            largest_comp = max(components, key=len)
            if len(largest_comp) < 2:
                return "Disconnected (no valid components)"
            subgraph = G.subgraph(largest_comp)
            try:
                return nx.diameter(subgraph)
            except nx.NetworkXError:
                return "Disconnected (no paths in largest component)"
        else:
            if nx.is_connected(G.to_undirected(as_view=True)):
                return nx.diameter(G)
            largest_comp = max(nx.connected_components(G.to_undirected(as_view=True)), key=len)
            subgraph = G.subgraph(largest_comp)
            try:
                return nx.diameter(subgraph)
            except nx.NetworkXError:
                return "Disconnected (no paths in largest component)"

    def _calculate_avg_path_length(self, G):
        """Calculate average path length, handling disconnected components"""
        if self.directed:
            components = list(nx.strongly_connected_components(G))
            avg_lengths = []
            for comp in components:
                if len(comp) > 1:
                    subgraph = G.subgraph(comp)
                    try:
                        avg_lengths.append(nx.average_shortest_path_length(subgraph))
                    except nx.NetworkXError:
                        continue
            if avg_lengths:
                return f"Disconnected (avg: {np.mean(avg_lengths):.2f})"
            return "Disconnected (no valid components)"
        else:
            if nx.is_connected(G.to_undirected(as_view=True)):
                return nx.average_shortest_path_length(G)
            components = list(nx.connected_components(G.to_undirected(as_view=True)))
            avg_lengths = []
            for comp in components:
                if len(comp) > 1:
                    subgraph = G.subgraph(comp)
                    try:
                        avg_lengths.append(nx.average_shortest_path_length(subgraph))
                    except nx.NetworkXError:
                        continue
            if avg_lengths:
                return f"Disconnected (avg: {np.mean(avg_lengths):.2f})"
            return "Disconnected (no valid components)"

    def detect_communities(self, b):
        if self.G is None:
            print("Please load a graph first")
            return

        with self.community_output:
            clear_output(wait=True)
            algorithm = self.community_dropdown.value
            self.community_results = {}
            working_graph = self.G
            was_directed = self.directed

            if was_directed:
                print("Warning: Community detection on directed graphs may differ from undirected.")
                print("Consider converting to undirected for analysis.")

            print("Starting community detection...")
            if algorithm in ["Louvain", "All"]:
                print("Running Louvain algorithm...")
                try:
                    undirected_graph = working_graph.to_undirected(as_view=True)
                    partition = community_louvain.best_partition(undirected_graph)
                    communities = defaultdict(list)
                    for node, community_id in partition.items():
                        communities[community_id].append(node)

                    modularity = (
                        self._directed_modularity(partition, self.G) if was_directed else
                        community_louvain.modularity(partition, undirected_graph)
                    )
                    conductance = self._calculate_conductance(undirected_graph, list(communities.values()))
                    silhouette = self._calculate_silhouette(undirected_graph, partition)

                    self.community_results['Louvain'] = {
                        'communities': list(communities.values()),
                        'modularity': modularity,
                        'coverage': len(communities) / undirected_graph.number_of_nodes(),
                        'partition': partition,
                        'conductance': conductance,
                        'silhouette': silhouette
                    }

                    for node, comm in partition.items():
                        self.G.nodes[node]['community'] = comm
                        self.G.nodes[node]['louvain_community'] = comm

                except Exception as e:
                    print(f"Error in Louvain algorithm: {str(e)}")
                    if algorithm != "All":
                        return

            if algorithm in ["Girvan-Newman", "All"]:
                print("\nRunning Girvan-Newman algorithm...")
                try:
                    G_copy = working_graph.copy()
                    total_edges = G_copy.number_of_edges()
                    pbar = tqdm(total=total_edges, desc="Removing edges", unit="edge")

                    def most_valuable_edge(G):
                        centrality = nx.edge_betweenness_centrality(G, normalized=True)
                        edge = max(centrality.items(), key=lambda x: x[1])[0]
                        pbar.update(1)
                        return edge

                    communities_generator = nx.algorithms.community.girvan_newman(G_copy, most_valuable_edge=most_valuable_edge)
                    target_communities = 6
                    gn_communities = None
                    for _ in range(target_communities - 1):
                        gn_communities = next(communities_generator)

                    pbar.close()
                    gn_partition = {node: i for i, comm in enumerate(gn_communities) for node in comm}
                    modularity = (
                        self._directed_modularity(gn_partition, self.G) if was_directed else
                        community_louvain.modularity(gn_partition, working_graph)
                    )
                    conductance = self._calculate_conductance(working_graph, gn_communities)
                    silhouette = self._calculate_silhouette(working_graph, gn_partition)

                    self.community_results['Girvan-Newman'] = {
                        'communities': gn_communities,
                        'modularity': modularity,
                        'coverage': len(gn_communities) / working_graph.number_of_nodes(),
                        'partition': gn_partition,
                        'conductance': conductance,
                        'silhouette': silhouette
                    }

                    for node, comm in gn_partition.items():
                        self.G.nodes[node]['community'] = comm
                        self.G.nodes[node]['gn_community'] = comm

                    print(f"Girvan-Newman completed. Found {len(gn_communities)} communities.")

                except Exception as e:
                    print(f"Error in Girvan-Newman: {str(e)}")
                    pbar.close()
                    if algorithm != "All":
                        return

            print("Community detection completed!\n")
            self.node_color_dropdown.value = 'uniform'
            self.draw_graph()

            print("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
                'Algorithm', '# Communities', 'Modularity', 'Coverage', 'Conductance', 'Silhouette'))
            print("-" * 95)
            for algo, results in self.community_results.items():
                print("{:<20} {:<15} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                    algo, len(results['communities']), results['modularity'],
                    results['coverage'], results['conductance'], results['silhouette']
          ))

            if algorithm == "All" and len(self.community_results) >= 2:
                print("\nCommunity Detection Comparison:")
                algorithms = list(self.community_results.keys())
                for algo1, algo2 in combinations(algorithms, 2):
                    part1 = self.community_results[algo1]['partition']
                    part2 = self.community_results[algo2]['partition']
                    common_nodes = set(part1.keys()) & set(part2.keys())
                    part1_values = [part1[n] for n in common_nodes]
                    part2_values = [part2[n] for n in common_nodes]
                    nmi = normalized_mutual_info_score(part1_values, part2_values)
                    ari = self._calculate_adjusted_rand_index(part1_values, part2_values)
                    vi = self._calculate_variation_of_information(part1_values, part2_values)
                    print(f"\nComparison: {algo1} vs {algo2}")
                    print(f"Normalized Mutual Info: {nmi:.4f}")
                    print(f"Adjusted Rand Index: {ari:.4f}")
                    print(f"Variation of Information: {vi:.4f}")

    def _calculate_conductance(self, G, communities):
        if len(communities) <= 1 or not communities:
            return 0.0

        total_conductance = 0
        for community in communities:
            if not community:
                continue
            community_set = set(community)
            internal_edges = sum(1 for node in community for neighbor in G.neighbors(node) if neighbor in community_set)
            external_edges = sum(1 for node in community for neighbor in G.neighbors(node) if neighbor not in community_set)
            if internal_edges + external_edges == 0:
                community_conductance = 0
            else:
                community_conductance = external_edges / (internal_edges + external_edges)
            total_conductance += community_conductance
        return total_conductance / len(communities)

    def _calculate_silhouette(self, G, partition):
        if len(set(partition.values())) <= 1 or not partition:
            return 0.0

        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)

        if len(communities) <= 1:
            return 0

        nodes = list(G.nodes())
        n = len(nodes)
        if n > 200:
            sample_size = min(200, n)
            sample_nodes = np.random.choice(nodes, sample_size, replace=False)
            sampled_partition = {node: partition[node] for node in sample_nodes if node in partition}
            return self._calculate_silhouette_sample(G, sampled_partition)

        total_silhouette = 0
        count = 0
        for node, community_id in partition.items():
            own_community = communities[community_id]
            if len(own_community) <= 1:
                continue

            own_distances = []
            for other_node in own_community:
                if other_node != node:
                    try:
                        dist = nx.shortest_path_length(G, node, other_node)
                        own_distances.append(dist)
                    except nx.NetworkXNoPath:
                        own_distances.append(float('inf'))

            if not own_distances or all(d == float('inf') for d in own_distances):
                continue

            a_i = np.mean([d for d in own_distances if d != float('inf')])
            b_i = float('inf')
            for other_comm_id, other_comm_nodes in communities.items():
                if other_comm_id == community_id:
                    continue
                other_distances = []
                for other_node in other_comm_nodes:
                    try:
                        dist = nx.shortest_path_length(G, node, other_node)
                        other_distances.append(dist)
                    except nx.NetworkXNoPath:
                        other_distances.append(float('inf'))
                if other_distances and not all(d == float('inf') for d in other_distances):
                    avg_dist = np.mean([d for d in other_distances if d != float('inf')])
                    b_i = min(b_i, avg_dist)

            if b_i == float('inf'):
                continue

            s_i = (b_i - a_i) / max(a_i, b_i)
            total_silhouette += s_i
            count += 1

        return total_silhouette / count if count > 0 else 0

    def _calculate_silhouette_sample(self, G, partition):
        if not partition:
            return 0

        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)

        total_silhouette = 0
        count = 0
        for node, community_id in partition.items():
            own_community = [n for n in communities[community_id] if n in partition]
            if len(own_community) <= 1:
                continue

            own_distances = []
            for other_node in own_community:
                if other_node != node:
                    try:
                        dist = nx.shortest_path_length(G, node, other_node)
                        own_distances.append(dist)
                    except nx.NetworkXNoPath:
                        own_distances.append(float('inf'))

            if not own_distances or all(d == float('inf') for d in own_distances):
                continue

            a_i = np.mean([d for d in own_distances if d != float('inf')])
            b_i = float('inf')
            for other_comm_id in communities:
                if other_comm_id == community_id:
                    continue
                other_comm_nodes = [n for n in communities[other_comm_id] if n in partition]
                other_distances = []
                for other_node in other_comm_nodes:
                    try:
                        dist = nx.shortest_path_length(G, node, other_node)
                        other_distances.append(dist)
                    except nx.NetworkXNoPath:
                        other_distances.append(float('inf'))
                if other_distances and not all(d == float('inf') for d in other_distances):
                    avg_dist = np.mean([d for d in other_distances if d != float('inf')])
                    b_i = min(b_i, avg_dist)

            if b_i == float('inf'):
                continue

            s_i = (b_i - a_i) / max(a_i, b_i)
            total_silhouette += s_i
            count += 1

        return total_silhouette / count if count > 0 else 0

    def _calculate_adjusted_rand_index(self, labels1, labels2):
        if not labels1 or not labels2 or len(labels1) != len(labels2):
            return 0

        labels1 = np.array(labels1)
        labels2 = np.array(labels2)
        n = len(labels1)
        contingency = {}
        for i in range(n):
            pair = (labels1[i], labels2[i])
            contingency[pair] = contingency.get(pair, 0) + 1

        sum_comb = sum(count * (count - 1) // 2 for count in contingency.values())
        sum1 = sum(count * (count - 1) // 2 for label in set(labels1) for count in [np.sum(labels1 == label)])
        sum2 = sum(count * (count - 1) // 2 for label in set(labels2) for count in [np.sum(labels2 == label)])
        total_comb = n * (n - 1) // 2
        expected_index = (sum1 * sum2) / total_comb
        max_index = (sum1 + sum2) / 2

        return (sum_comb - expected_index) / (max_index - expected_index) if max_index != expected_index else 0

    def _calculate_variation_of_information(self, labels1, labels2):
        if not labels1 or not labels2 or len(labels1) != len(labels2):
            return float('inf')

        labels1 = np.array(labels1)
        labels2 = np.array(labels2)
        n = len(labels1)

        entropy1 = -sum(p * np.log2(p) for label in set(labels1) for p in [np.sum(labels1 == label) / n] if p > 0)
        entropy2 = -sum(p * np.log2(p) for label in set(labels2) for p in [np.sum(labels2 == label) / n] if p > 0)
        mi = 0
        for label1 in set(labels1):
            for label2 in set(labels2):
                joint_p = np.sum((labels1 == label1) & (labels2 == label2)) / n
                if joint_p > 0:
                    p1 = np.sum(labels1 == label1) / n
                    p2 = np.sum(labels2 == label2) / n
                    mi += joint_p * np.log2(joint_p / (p1 * p2))
        return entropy1 + entropy2 - 2 * mi

    def _directed_modularity(self, partition, G):
        m = G.size(weight='weight') if G.edges() and 'weight' in next(iter(G.edges(data=True)))[2] else G.size()
        if m == 0:
            return 0

        total = 0.0
        for community in set(partition.values()):
            nodes_in_comm = [n for n in G.nodes() if partition[n] == community]
            l_c = sum(G.in_degree(n, weight='weight') * G.out_degree(n, weight='weight') for n in nodes_in_comm)
            k_c_in = sum(G.in_degree(n, weight='weight') for n in nodes_in_comm)
            k_c_out = sum(G.out_degree(n, weight='weight') for n in nodes_in_comm)
            total += (l_c / m) - (k_c_in * k_c_out) / (m * m)
        return total

    def link_analysis(self, b):
        if self.G is None:
            print("Please load a graph first")
            return

        with self.link_output:
            clear_output(wait=True)
            self.centrality_measures['Degree'] = nx.degree_centrality(self.G)
            self.centrality_measures['Betweenness'] = nx.betweenness_centrality(self.G)
            self.centrality_measures['Closeness'] = nx.closeness_centrality(self.G)
            try:
                self.centrality_measures['Eigenvector'] = nx.eigenvector_centrality(self.G, max_iter=1000)
            except:
                if self.directed:
                    print("Warning: Using scaled eigenvector centrality for directed graph")
                    self.centrality_measures['Eigenvector'] = nx.eigenvector_centrality_numpy(self.G)
                else:
                    raise
            self.centrality_measures['PageRank'] = nx.pagerank(self.G)

            self.centrality_ranges = {}
            for measure_name, measure in self.centrality_measures.items():
                values = list(measure.values())
                self.centrality_ranges[measure_name] = {'min': min(values), 'max': max(values)}

            print("Top Nodes by Centrality Measures:\n")
            for measure_name, measure in self.centrality_measures.items():
                print(f"\n{measure_name} Centrality:")
                print(f"Range: {self.centrality_ranges[measure_name]['min']:.6f} to {self.centrality_ranges[measure_name]['max']:.6f}")
                top_nodes = sorted(measure.items(), key=lambda x: x[1], reverse=True)[:10]
                print("{:<15} {:<15}".format('Node', 'Score'))
                print("-" * 30)
                for node, score in top_nodes:
                    print("{:<15} {:<15.6f}".format(str(node), score))

tool = EnhancedNetworkAnalysisTool()