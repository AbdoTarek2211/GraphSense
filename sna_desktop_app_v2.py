import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import time
import warnings
import psutil  # For resource monitoring
from collections import defaultdict
from community import community_louvain
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from itertools import combinations
from networkx.algorithms import community as nx_community
from PyQt5.QtWidgets import (
    QSizePolicy, QProgressDialog, QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog,
    QTabWidget, QTextEdit, QGroupBox, QDoubleSpinBox, QScrollArea, QColorDialog,
    QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QTextCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

warnings.filterwarnings('ignore')

class EnhancedNetworkAnalysisTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.G = None
        self.directed = False
        self.node_attributes = {}
        self.edge_attributes = {}
        self.current_layout = 'spring'
        self.community_results = {}
        self.centrality_measures = {}
        self.filtered_graph = None
        self.centrality_ranges = {}
        self.node_color = 'skyblue'  # Default node color
        self.edge_color = 'gray'     # Default edge color
        self.scale_factor = 1.0      # Default scaling factor for visualization
        self.event_cids = []         # Track Matplotlib event connections

        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Initialize widgets
        self.init_widgets()

        # Window settings
        self.setWindowTitle("Network Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.show()

    def init_widgets(self):
        """Initialize all PyQt widgets with controls in Graph tab and graph display in Visualization tab"""
        # Create tab widget for different sections
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # Graph tab (controls only)
        self.graph_tab = QWidget()
        self.tab_widget.addTab(self.graph_tab, "Graph")
        self.graph_layout = QVBoxLayout(self.graph_tab)

        # Control panel for graph settings
        control_panel = QGroupBox("Graph Controls")
        control_layout = QVBoxLayout()

        # Row 1 - graph type, show labels, and file loading
        row1 = QHBoxLayout()
        self.directed_checkbox = QCheckBox("Directed Graph")
        row1.addWidget(self.directed_checkbox)

        self.show_labels_checkbox = QCheckBox("Show labels")
        row1.addWidget(self.show_labels_checkbox)

        self.load_nodes_button = QPushButton("Load Nodes CSV")
        self.load_nodes_button.clicked.connect(self.load_nodes)
        row1.addWidget(self.load_nodes_button)

        self.load_edges_button = QPushButton("Load Edges CSV")
        self.load_edges_button.clicked.connect(self.load_edges)
        row1.addWidget(self.load_edges_button)
        control_layout.addLayout(row1)

        # Row 2 - color pickers (aligned)
        row2 = QHBoxLayout()
        self.node_color_label = QLabel("Node color:")
        self.node_color_label.setFixedWidth(100)
        row2.addWidget(self.node_color_label)
        self.node_color_display = QLabel()
        self.node_color_display.setFixedWidth(150)
        self.node_color_display.setStyleSheet(f"background-color: {self.node_color}; border: 1px solid black;")
        row2.addWidget(self.node_color_display)
        self.node_color_button = QPushButton("Pick Node Color")
        self.node_color_button.clicked.connect(self.pick_node_color)
        row2.addWidget(self.node_color_button)

        self.edge_color_label = QLabel("Edge color:")
        self.edge_color_label.setFixedWidth(100)
        row2.addWidget(self.edge_color_label)
        self.edge_color_display = QLabel()
        self.edge_color_display.setFixedWidth(150)
        self.edge_color_display.setStyleSheet(f"background-color: {self.edge_color}; border: 1px solid black;")
        row2.addWidget(self.edge_color_display)
        self.edge_color_button = QPushButton("Pick Edge Color")
        self.edge_color_button.clicked.connect(self.pick_edge_color)
        row2.addWidget(self.edge_color_button)
        row2.addStretch()
        control_layout.addLayout(row2)

        # Row 3 - layout and node size (aligned)
        row3 = QHBoxLayout()
        self.layout_label = QLabel("Layout:")
        self.layout_label.setFixedWidth(100)
        row3.addWidget(self.layout_label)
        self.layout_dropdown = QComboBox()
        self.layout_dropdown.setFixedWidth(150)
        self.layout_dropdown.addItems([
            'spring', 'circular', 'random', 'shell', 'spectral',
            'kamada_kawai', 'tree', 'radial', 'stress', 'multipartite'
        ])
        row3.addWidget(self.layout_dropdown)

        self.node_size_label = QLabel("Node size by:")
        self.node_size_label.setFixedWidth(100)
        row3.addWidget(self.node_size_label)
        self.node_size_dropdown = QComboBox()
        self.node_size_dropdown.setFixedWidth(150)
        self.node_size_dropdown.addItem("uniform")
        row3.addWidget(self.node_size_dropdown)
        row3.addStretch()
        control_layout.addLayout(row3)

        # Row 4 - node color by and node shape (aligned)
        row4 = QHBoxLayout()
        self.node_color_attr_label = QLabel("Node color by:")
        self.node_color_attr_label.setFixedWidth(100)
        row4.addWidget(self.node_color_attr_label)
        self.node_color_dropdown = QComboBox()
        self.node_color_dropdown.setFixedWidth(150)
        self.node_color_dropdown.addItem("uniform")
        row4.addWidget(self.node_color_dropdown)

        self.node_shape_label = QLabel("Node shape:")
        self.node_shape_label.setFixedWidth(100)
        row4.addWidget(self.node_shape_label)
        self.node_shape_dropdown = QComboBox()
        self.node_shape_dropdown.setFixedWidth(150)
        self.node_shape_dropdown.addItems(['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', '8'])
        row4.addWidget(self.node_shape_dropdown)
        row4.addStretch()
        control_layout.addLayout(row4)

        # Row 5 - filtering and scaling (aligned)
        row5 = QHBoxLayout()
        self.filter_label = QLabel("Filter by:")
        self.filter_label.setFixedWidth(100)
        row5.addWidget(self.filter_label)
        self.filter_dropdown = QComboBox()
        self.filter_dropdown.setFixedWidth(150)
        self.filter_dropdown.addItems([
            "None", "Degree Centrality", "Betweenness Centrality",
            "Closeness Centrality", "Eigenvector Centrality", "PageRank"
        ])
        self.filter_dropdown.currentTextChanged.connect(self.on_filter_change)
        row5.addWidget(self.filter_dropdown)

        self.scale_label = QLabel("Scale factor:")
        self.scale_label.setFixedWidth(100)
        row5.addWidget(self.scale_label)
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setFixedWidth(150)
        self.scale_spinbox.setRange(0.1, 5.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.setDecimals(2)
        self.scale_spinbox.valueChanged.connect(self.on_scale_change)
        row5.addWidget(self.scale_spinbox)
        row5.addStretch()
        control_layout.addLayout(row5)

        # Row 6 - community algorithm and Girvan-Newman target communities (aligned)
        row6 = QHBoxLayout()
        self.community_label = QLabel("Community Algorithm:")
        self.community_label.setFixedWidth(100)
        row6.addWidget(self.community_label)
        self.community_dropdown = QComboBox()
        self.community_dropdown.setFixedWidth(150)
        self.community_dropdown.addItems(["Louvain", "Girvan-Newman", "All"])
        row6.addWidget(self.community_dropdown)

        self.gn_target_label = QLabel("Target Communities:")
        self.gn_target_label.setFixedWidth(100)
        row6.addWidget(self.gn_target_label)
        self.gn_target_spinbox = QSpinBox()
        self.gn_target_spinbox.setFixedWidth(150)
        self.gn_target_spinbox.setRange(1, 50)  # Reasonable range for community count
        self.gn_target_spinbox.setValue(4)  # Default to match original code
        row6.addWidget(self.gn_target_spinbox)
        row6.addStretch()
        control_layout.addLayout(row6)

        # Row 7 - sliders (aligned)
        row7 = QHBoxLayout()
        self.min_slider_label = QLabel("Min threshold:")
        self.min_slider_label.setFixedWidth(100)
        row7.addWidget(self.min_slider_label)
        self.min_slider = QDoubleSpinBox()
        self.min_slider.setFixedWidth(150)
        self.min_slider.setRange(0, 1)
        self.min_slider.setSingleStep(0.01)
        self.min_slider.setValue(0)
        self.min_slider.setDecimals(4)
        row7.addWidget(self.min_slider)

        self.max_slider_label = QLabel("Max threshold:")
        self.max_slider_label.setFixedWidth(100)
        row7.addWidget(self.max_slider_label)
        self.max_slider = QDoubleSpinBox()
        self.max_slider.setFixedWidth(150)
        self.max_slider.setRange(0, 1)
        self.max_slider.setSingleStep(0.01)
        self.max_slider.setValue(1)
        self.max_slider.setDecimals(4)
        row7.addWidget(self.max_slider)
        row7.addStretch()
        control_layout.addLayout(row7)

        # Row 8 - buttons
        row8 = QHBoxLayout()
        self.reset_filter_button = QPushButton("Reset Filters")
        self.reset_filter_button.clicked.connect(self.reset_filters)
        row8.addWidget(self.reset_filter_button)

        self.reset_sliders_button = QPushButton("Reset Sliders")
        self.reset_sliders_button.clicked.connect(lambda: self.reset_sliders())
        self.reset_sliders_button.setVisible(False)
        row8.addWidget(self.reset_sliders_button)

        self.load_button = QPushButton("Load Graph")
        self.load_button.clicked.connect(self.load_graph)
        row8.addWidget(self.load_button)

        self.metrics_button = QPushButton("Calculate Metrics")
        self.metrics_button.clicked.connect(self.show_metrics)
        row8.addWidget(self.metrics_button)

        self.community_button = QPushButton("Detect Communities")
        self.community_button.clicked.connect(self.detect_communities)
        row8.addWidget(self.community_button)

        self.link_button = QPushButton("Link Analysis")
        self.link_button.clicked.connect(self.link_analysis)
        row8.addWidget(self.link_button)

        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.clicked.connect(self.apply_filter)
        row8.addWidget(self.apply_filter_button)
        control_layout.addLayout(row8)

        control_panel.setLayout(control_layout)
        self.graph_layout.addWidget(control_panel)

        # Visualization tab
        self.visualization_tab = QWidget()
        self.tab_widget.addTab(self.visualization_tab, "Visualization")
        self.visualization_layout = QVBoxLayout(self.visualization_tab)

        # Matplotlib figure for graph display
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Add navigation toolbar for zooming and panning
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.visualization_layout.addWidget(self.toolbar)

        # Create a scroll area for the canvas, ensuring it expands to fill the tab
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)  # Allow the canvas to resize with the scroll area
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Add the scroll area to the visualization tab, ensuring it takes all available space
        self.visualization_layout.addWidget(self.scroll_area, stretch=1)

        # Connect resize event to redraw the graph when the scroll area changes size
        self.scroll_area.viewport().resizeEvent = self.on_resize

        # Create other tabs
        self.create_metrics_tab()
        self.create_community_tab()
        self.create_link_analysis_tab()

    def detect_communities(self):
        """Detect communities using selected algorithm with resource checks"""
        if self.G is None:
            self.community_output.append("Please load a graph first")
            return

        # Clear the output at the start
        self.community_output.clear()
        algorithm = self.community_dropdown.currentText()
        self.community_results = {}
        working_graph = self.G
        was_directed = self.directed

        if was_directed:
            self.community_output.append(
                "\nWarning: Community detection on directed graphs can produce different results\n"
                "than on undirected graphs. Some metrics may not be directly comparable.\n"
                "Consider converting to undirected if appropriate for your analysis.\n"
            )

        self.community_output.append("Starting community detection...")
        QApplication.processEvents()

        # Check resources before running heavy computations
        if algorithm in ["Girvan-Newman", "All"]:
            resource_ok, resource_msg = self._check_resources()
            if not resource_ok:
                self.community_output.append(
                    f"Warning: {resource_msg}\n"
                    "Girvan-Newman algorithm is resource-intensive and may crash on low-resource systems.\n"
                    "Consider using Louvain algorithm instead or reducing graph size."
                )

        if algorithm in ["Louvain", "All"]:
            self.community_output.append("Running Louvain algorithm...")
            QApplication.processEvents()
            progress = QProgressDialog("Detecting communities with Louvain...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Community Detection")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            try:
                undirected_graph = working_graph.to_undirected(as_view=True)
                progress.setValue(20)
                QApplication.processEvents()
                partition = community_louvain.best_partition(undirected_graph)
                progress.setValue(60)
                QApplication.processEvents()
                communities = defaultdict(list)
                for node, community_id in partition.items():
                    communities[community_id].append(node)
                communities_list = list(communities.values())
                progress.setValue(80)
                QApplication.processEvents()
                louvain_modularity = (
                    self._directed_modularity(partition, self.G) if was_directed
                    else community_louvain.modularity(partition, undirected_graph)
                )
                conductance = self._calculate_conductance(undirected_graph, communities_list)
                silhouette = self._calculate_silhouette(undirected_graph, partition)
                self.community_results['Louvain'] = {
                    'communities': communities_list,
                    'modularity': louvain_modularity,
                    'coverage': len(communities) / undirected_graph.number_of_nodes(),
                    'partition': partition,
                    'conductance': conductance,
                    'silhouette': silhouette
                }
                for node, comm in partition.items():
                    self.G.nodes[node]['community'] = comm
                    self.G.nodes[node]['louvain_community'] = comm
                progress.setValue(100)
                progress.close()
            except Exception as e:
                progress.close()
                self.community_output.append(f"Error in Louvain algorithm: {str(e)}")

        if algorithm in ["Girvan-Newman", "All"]:
            self.community_output.append(f"\nRunning Girvan-Newman algorithm with target {self.gn_target_spinbox.value()} communities (this may take a while)...")
            QApplication.processEvents()
            progress = QProgressDialog("Detecting communities with Girvan-Newman...", "Cancel", 0, working_graph.number_of_edges(), self)
            progress.setWindowTitle("Community Detection")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            def update_progress(step):
                progress.setValue(progress.value() + step)
                QApplication.processEvents()
                return not progress.wasCanceled()

            try:
                G_copy = working_graph.copy()
                def most_valuable_edge(G):
                    centrality = nx.edge_betweenness_centrality(G, normalized=True)
                    edge = max(centrality.items(), key=lambda x: x[1])[0]
                    update_progress(1)
                    return edge
                communities_generator = nx.algorithms.community.girvan_newman(
                    G_copy, most_valuable_edge=most_valuable_edge
                )
                target_communities = self.gn_target_spinbox.value()
                if target_communities < 1 or target_communities > working_graph.number_of_nodes():
                    raise ValueError(f"Target communities must be between 1 and {working_graph.number_of_nodes()}")
                gn_communities = None
                for _ in range(target_communities - 1):
                    gn_communities = next(communities_generator, None)
                    if gn_communities is None:
                        raise ValueError(f"Cannot split into {target_communities} communities; graph fully disconnected")
                progress.close()
                if gn_communities is None:
                    raise ValueError("Girvan-Newman failed to produce communities")
                gn_partition = {node: i for i, comm in enumerate(gn_communities) for node in comm}
                gn_modularity = (
                    self._directed_modularity(gn_partition, self.G) if was_directed
                    else community_louvain.modularity(gn_partition, working_graph)
                )
                conductance = self._calculate_conductance(working_graph, gn_communities)
                silhouette = self._calculate_silhouette(working_graph, gn_partition)
                self.community_results['Girvan-Newman'] = {
                    'communities': gn_communities,
                    'modularity': gn_modularity,
                    'coverage': len(gn_communities) / working_graph.number_of_nodes(),
                    'partition': gn_partition,
                    'conductance': conductance,
                    'silhouette': silhouette
                }
                for node, comm in gn_partition.items():
                    self.G.nodes[node]['community'] = comm
                    self.G.nodes[node]['gn_community'] = comm
                self.community_output.append(f"Girvan-Newman completed. Found {len(gn_communities)} communities.")
            except Exception as e:
                progress.close()
                self.community_output.append(f"Error in Girvan-Newman: {str(e)}")

        self.community_output.append("\nCommunity detection completed!\n")
        self.community_output.append("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            'Algorithm', '# Communities', 'Modularity', 'Coverage', 'Conductance', 'Silhouette'
        ))
        self.community_output.append("-" * 95)
        for algo, results in self.community_results.items():
            self.community_output.append("{:<20} {:<15} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                algo, len(results['communities']), results['modularity'],
                results['coverage'], results['conductance'], results['silhouette']
            ))

        if algorithm == "All" and len(self.community_results) >= 2:
            self.community_output.append("\nCommunity Detection Comparison:")
            algorithms = list(self.community_results.keys())
            for algo1, algo2 in combinations(algorithms, 2):
                try:
                    part1 = self.community_results[algo1]['partition']
                    part2 = self.community_results[algo2]['partition']
                    common_nodes = set(part1.keys()) & set(part2.keys())
                    part1_values = [part1[n] for n in common_nodes]
                    part2_values = [part2[n] for n in common_nodes]
                    nmi = normalized_mutual_info_score(part1_values, part2_values)
                    ari = adjusted_rand_score(part1_values, part2_values)
                    vi = self._calculate_variation_of_information(part1_values, part2_values)
                    self.community_output.append(f"\nComparison: {algo1} vs {algo2}")
                    self.community_output.append(f"Normalized Mutual Info: {nmi:.4f}")
                    self.community_output.append(f"Adjusted Rand Index: {ari:.4f}")
                    self.community_output.append(f"Variation of Information: {vi:.4f}")
                except Exception as e:
                    self.community_output.append(f"\nComparison: {algo1} vs {algo2}")
                    self.community_output.append(f"Error during comparison: {str(e)}")

        self.node_color_dropdown.setCurrentText('uniform')
        self.draw_graph()
        self.community_output.moveCursor(QTextCursor.Start)

    def on_resize(self, event):
        """Handle resize events to redraw the graph to fit the new size"""
        if self.G is not None:
            self.draw_graph()

    def on_scale_change(self):
        """Handle changes to the scale factor"""
        self.scale_factor = self.scale_spinbox.value()
        self.draw_graph()

    def pick_node_color(self):
        """Open a color picker dialog for node color"""
        color = QColorDialog.getColor(initial=QColor(self.node_color), parent=self, title="Select Node Color")
        if color.isValid():
            self.node_color = color.name()
            self.node_color_display.setStyleSheet(f"background-color: {self.node_color}; border: 1px solid black;")
            self.draw_graph()

    def pick_edge_color(self):
        """Open a color picker dialog for edge color"""
        color = QColorDialog.getColor(initial=QColor(self.edge_color), parent=self, title="Select Edge Color")
        if color.isValid():
            self.edge_color = color.name()
            self.edge_color_display.setStyleSheet(f"background-color: {self.edge_color}; border: 1px solid black;")
            self.draw_graph()

    def create_metrics_tab(self):
        """Create the metrics display tab"""
        self.metrics_tab = QWidget()
        self.tab_widget.addTab(self.metrics_tab, "Metrics")
        metrics_layout = QVBoxLayout(self.metrics_tab)
        self.metrics_output = QTextEdit()
        self.metrics_output.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_output)

    def create_community_tab(self):
        """Create the community analysis tab"""
        self.community_tab = QWidget()
        self.tab_widget.addTab(self.community_tab, "Communities")
        community_layout = QVBoxLayout(self.community_tab)
        self.community_output = QTextEdit()
        self.community_output.setReadOnly(True)
        community_layout.addWidget(self.community_output)

    def create_link_analysis_tab(self):
        """Create the link analysis tab"""
        self.link_tab = QWidget()
        self.tab_widget.addTab(self.link_tab, "Link Analysis")
        link_layout = QVBoxLayout(self.link_tab)
        self.link_output = QTextEdit()
        self.link_output.setReadOnly(True)
        link_layout.addWidget(self.link_output)

    def on_filter_change(self, filter_type):
        """Handle filter dropdown changes with error recovery"""
        try:
            self.update_filter_options(filter_type)
            self.reset_sliders_button.setVisible(False)
        except Exception as e:
            self.metrics_output.append(f"Error updating filter: {str(e)}")
            self.reset_sliders_button.setVisible(True)

    def _safe_update_single_slider(self, slider, new_min, new_max, new_value):
        """Safely update a single slider's attributes"""
        try:
            if new_min >= new_max:
                new_max = new_min + max(new_min * 0.1, 0.001)
            safe_value = max(new_min, min(new_value, new_max))
            slider.setMinimum(new_min)
            slider.setMaximum(new_max)
            slider.setValue(safe_value)
            return True
        except Exception as e:
            self.metrics_output.append(f"Error updating slider: {str(e)}")
            self.reset_sliders_button.setVisible(True)
            return False

    def reset_sliders(self, silent=False):
        """Completely reset sliders to default state"""
        try:
            self.min_slider.setDecimals(4)
            self.max_slider.setDecimals(4)
            self.min_slider.setSingleStep(0.01)
            self.max_slider.setSingleStep(0.01)
            self._safe_update_single_slider(self.min_slider, 0, 1, 0)
            self._safe_update_single_slider(self.max_slider, 0, 1, 1)
            self.min_slider_label.setText('Min threshold:')
            self.max_slider_label.setText('Max threshold:')
            if not silent:
                self.metrics_output.append("Sliders reset successfully")
            self.reset_sliders_button.setVisible(False)
        except Exception as e:
            if not silent:
                self.metrics_output.append(f"Error resetting sliders: {str(e)}")
            self.reset_sliders_button.setVisible(True)

    def update_filter_options(self, filter_type):
        """Update slider ranges and options based on the selected filter"""
        self.reset_sliders(silent=True)

        if filter_type == "None":
            self._safe_update_single_slider(self.min_slider, 0, 1, 0)
            self._safe_update_single_slider(self.max_slider, 0, 1, 1)
            return

        if filter_type == "Community":
            if not self.community_results:
                self.metrics_output.append("Please detect communities first")
                return
            algorithm = self.community_dropdown.currentText()
            if algorithm in self.community_results:
                communities = self.community_results[algorithm]['communities']
                num_communities = len(communities)
                if num_communities > 0:
                    self.min_slider_label.setText('Min community:')
                    self.max_slider_label.setText('Max community:')
                    self.min_slider.setDecimals(0)
                    self.max_slider.setDecimals(0)
                    self.min_slider.setSingleStep(1)
                    self.max_slider.setSingleStep(1)
                    self._safe_update_single_slider(self.min_slider, 0, num_communities-1, 0)
                    self._safe_update_single_slider(self.max_slider, 0, num_communities-1, num_communities-1)
                else:
                    self.metrics_output.append("No communities found")
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
                self.metrics_output.append(f"Please run Link Analysis to calculate {filter_type} first")
                return
            min_val = self.centrality_ranges[centrality_name]['min']
            max_val = self.centrality_ranges[centrality_name]['max']

            if filter_type == "PageRank":
                decimals = 6
                step = 0.00001
            elif filter_type == "Betweenness Centrality":
                decimals = 6
                step = 0.00001
            elif filter_type == "Closeness Centrality":
                decimals = 6
                step = 0.0001
            else:
                decimals = 4
                step = 0.001

            if abs(max_val - min_val) < 1e-10:
                if min_val == 0:
                    min_val = 0
                    max_val = 0.1
                else:
                    min_val = max(0, min_val * 0.9)
                    max_val = min_val * 1.1

            buffer = max((max_val - min_val) * 0.01, 1e-10)
            min_val = max(0, min_val - buffer)
            max_val = max_val + buffer

            self.min_slider_label.setText('Min value:')
            self.max_slider_label.setText('Max value:')
            self.min_slider.setDecimals(decimals)
            self.max_slider.setDecimals(decimals)
            self.min_slider.setSingleStep(step)
            self.max_slider.setSingleStep(step)
            self._safe_update_single_slider(self.min_slider, min_val, max_val, min_val)
            self._safe_update_single_slider(self.max_slider, min_val, max_val, max_val)
        else:
            self.metrics_output.append(f"Please calculate {filter_type} first using Link Analysis")

    def reset_filters(self):
        """Reset all filters to default state"""
        self.filter_dropdown.setCurrentText("None")
        self.min_slider.setValue(self.min_slider.minimum())
        self.max_slider.setValue(self.max_slider.maximum())
        self.filtered_graph = None
        self.metrics_output.append("Filters reset. Showing full graph.")
        self.draw_graph()

    def apply_filter(self):
        """Apply selected filters to the graph"""
        if self.G is None:
            self.metrics_output.append("Please load a graph first")
            return

        filter_type = self.filter_dropdown.currentText()
        min_val = self.min_slider.value()
        max_val = self.max_slider.value()
        self.filtered_graph = None

        if filter_type == "None":
            self.draw_graph()
            return

        if filter_type == "Community":
            if not self.community_results:
                self.metrics_output.append("Please detect communities first")
                return
            algorithm = self.community_dropdown.currentText()
            if algorithm not in self.community_results:
                self.metrics_output.append(f"No communities detected with {algorithm} algorithm")
                return
            communities = self.community_results[algorithm]['communities']
            min_idx = max(0, min(int(min_val), len(communities)-1))
            max_idx = max(0, min(int(max_val), len(communities)-1))
            if min_idx > max_idx:
                min_idx, max_idx = max_idx, min_idx
            selected_communities = communities[min_idx:max_idx+1]
            if not selected_communities:
                self.metrics_output.append("No communities match the selected criteria. Try adjusting filters.")
                self.filtered_graph = self.G.subgraph([])
                self.draw_graph(self.filtered_graph)
                return
            nodes_to_keep = [node for comm in selected_communities for node in comm]
            self.filtered_graph = self.G.subgraph(nodes_to_keep)
        else:
            filter_to_centrality = {
                "Degree Centrality": "Degree",
                "Betweenness Centrality": "Betweenness",
                "Closeness Centrality": "Closeness",
                "Eigenvector Centrality": "Eigenvector",
                "PageRank": "PageRank"
            }
            centrality_name = filter_to_centrality.get(filter_type)
            if centrality_name not in self.centrality_measures:
                self.metrics_output.append(f"Please calculate {filter_type} first using Link Analysis")
                return
            centrality = self.centrality_measures[centrality_name]
            nodes_to_keep = [node for node, score in centrality.items() if min_val <= score <= max_val]
            if not nodes_to_keep:
                self.metrics_output.append("Filter removed all nodes. Try adjusting the threshold values.")
                self.filtered_graph = self.G.subgraph([])
                self.draw_graph(self.filtered_graph)
                return
            self.filtered_graph = self.G.subgraph(nodes_to_keep)

        if self.filtered_graph.number_of_nodes() > 0:
            self.metrics_output.append(
                f"Showing filtered graph with {self.filtered_graph.number_of_nodes()} "
                f"nodes and {self.filtered_graph.number_of_edges()} edges"
            )
        else:
            self.metrics_output.append("No nodes match the filter criteria")
        self.draw_graph(self.filtered_graph)

    def load_nodes(self):
        """Load nodes CSV file"""
        nodes_file, _ = QFileDialog.getOpenFileName(self, "Select Nodes CSV", "", "CSV Files (*.csv)")
        if nodes_file:
            try:
                self.node_df = pd.read_csv(nodes_file)
                if 'ID' not in self.node_df.columns:
                    raise ValueError("Nodes CSV must contain an 'ID' column")
                self.node_attributes = self.node_df.set_index('ID').to_dict('index')
                attributes = [col for col in self.node_df.columns if col != 'ID']
                self.node_size_dropdown.clear()
                self.node_size_dropdown.addItems(['uniform'] + attributes)
                self.node_color_dropdown.clear()
                self.node_color_dropdown.addItems(['uniform'] + attributes)
                self.metrics_output.append("Nodes CSV loaded successfully")
            except Exception as e:
                self.metrics_output.append(f"Error loading nodes file: {str(e)}")

    def load_edges(self):
        """Load edges CSV file"""
        edges_file, _ = QFileDialog.getOpenFileName(self, "Select Edges CSV", "", "CSV Files (*.csv)")
        if edges_file:
            try:
                self.edge_df = pd.read_csv(edges_file)
                self.edge_df.columns = self.edge_df.columns.str.lower()
                if not {'source', 'target'}.issubset(self.edge_df.columns):
                    raise ValueError("Edges CSV must contain 'source' and 'target' columns")
                if self.edge_df.duplicated(subset=['source', 'target']).any():
                    self.metrics_output.append("Warning: Duplicate edges found. Aggregating them...")
                    if 'weight' in self.edge_df.columns:
                        self.edge_df = self.edge_df.groupby(['source', 'target'])['weight'].sum().reset_index()
                    else:
                        self.edge_df = self.edge_df.drop_duplicates(subset=['source', 'target'])
                self.edge_attributes = self.edge_df.to_dict('records')
                self.metrics_output.append("Edges CSV loaded successfully")
            except Exception as e:
                self.metrics_output.append(f"Error loading edges file: {str(e)}")

    def load_graph(self):
        """Load and process graph data from loaded CSV files"""
        if not hasattr(self, 'node_df') or not hasattr(self, 'edge_df'):
            self.metrics_output.append("Please load both nodes and edges files")
            return

        try:
            self.directed = self.directed_checkbox.isChecked()
            self.G = nx.DiGraph() if self.directed else nx.Graph()

            for node, attrs in self.node_attributes.items():
                self.G.add_node(node, **attrs)

            for edge in self.edge_attributes:
                source = edge['source']
                target = edge['target']
                edge_attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
                self.G.add_edge(source, target, **edge_attrs)

            self.filtered_graph = None
            self.metrics_output.append("Graph loaded successfully!")
            self.draw_graph()
        except Exception as e:
            self.metrics_output.append(f"Error loading graph: {str(e)}")

    def _layout_for_disconnected_components(self, graph, layout_func, offset_increment=1.5):
        """Helper to handle layout for disconnected components"""
        if self.directed:
            components = list(nx.weakly_connected_components(graph))
        else:
            components = list(nx.connected_components(graph))
        pos = {}
        offset = 0
        for component in components:
            subgraph = graph.subgraph(component)
            if len(component) > 1:
                sub_pos = layout_func(subgraph)
                for node, coords in sub_pos.items():
                    pos[node] = coords + np.array([offset, 0])
            else:
                node = list(component)[0]
                pos[node] = np.array([offset, 0])
            offset += offset_increment
        return pos

    def draw_graph(self, graph=None):
        """Draw the graph visualization with dynamic sizing to fill the scroll area"""
        if graph is None:
            graph = self.G if self.filtered_graph is None else self.filtered_graph

        if graph is None or graph.number_of_nodes() == 0:
            self.metrics_output.append("No graph to display or graph is empty")
            self.figure.clear()
            self.canvas.draw()
            return

        # Disconnect previous event handlers to avoid accumulation
        for cid in self.event_cids:
            self.figure.canvas.mpl_disconnect(cid)
        self.event_cids = []

        # Get the available space from the scroll area's viewport
        viewport_size = self.scroll_area.viewport().size()
        available_width = viewport_size.width()
        available_height = viewport_size.height()

        # Calculate figure size in inches to match the viewport
        dpi = self.figure.get_dpi()
        fig_width = available_width / dpi
        fig_height = available_height / dpi

        # Set the figure size to fill the viewport
        self.figure.set_size_inches(fig_width, fig_height)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        #ax.set_aspect('equal', adjustable='box')

        if self.directed:
            self.metrics_output.append("Note: Directed graph - some layouts may not show directionality well")
            recommended_layouts = ['spring', 'multipartite', 'tree']
        else:
            recommended_layouts = [self.layout_dropdown.itemText(i) for i in range(self.layout_dropdown.count())]

        layout = self.layout_dropdown.currentText()
        if self.directed and layout not in recommended_layouts:
            self.metrics_output.append(f"Warning: {layout} layout may not work well with directed graphs")

        # Check if positions are already stored in the graph (e.g., from dragging)
        pos = nx.get_node_attributes(graph, 'pos')
        if not pos or layout != self.current_layout:
            self.current_layout = layout
            try:
                if layout == 'spring':
                    k_value = (1.0 / np.sqrt(graph.number_of_nodes()) + self.scale_factor * 0.1)
                    pos = nx.spring_layout(graph, k=k_value, iterations=200, seed=42)
                elif layout == 'circular':
                    pos = nx.circular_layout(graph)
                elif layout == 'random':
                    pos = nx.random_layout(graph, seed=42)
                elif layout == 'shell':
                    if self.directed:
                        if nx.is_weakly_connected(graph):
                            pos = nx.shell_layout(graph)
                        else:
                            pos = self._layout_for_disconnected_components(graph, nx.shell_layout)
                    else:
                        if nx.is_connected(graph):
                            pos = nx.shell_layout(graph)
                        else:
                            pos = self._layout_for_disconnected_components(graph, nx.shell_layout)
                elif layout == 'spectral':
                    if self.directed:
                        if nx.is_weakly_connected(graph):
                            pos = nx.spectral_layout(graph)
                        else:
                            pos = self._layout_for_disconnected_components(graph, nx.spectral_layout)
                    else:
                        if nx.is_connected(graph):
                            pos = nx.spectral_layout(graph)
                        else:
                            pos = self._layout_for_disconnected_components(graph, nx.spectral_layout)
                elif layout == 'kamada_kawai':
                    if self.directed:
                        if nx.is_weakly_connected(graph):
                            pos = nx.kamada_kawai_layout(graph)
                        else:
                            pos = self._layout_for_disconnected_components(graph, nx.kamada_kawai_layout)
                    else:
                        if nx.is_connected(graph):
                            pos = nx.kamada_kawai_layout(graph)
                        else:
                            pos = self._layout_for_disconnected_components(graph, nx.kamada_kawai_layout)
                elif layout == 'tree':
                    try:
                        if nx.is_tree(graph):
                            root = max(graph.degree(), key=lambda x: x[1])[0]
                            pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot', root=root)
                        else:
                            root = max(graph.degree(), key=lambda x: x[1])[0]
                            pos = self._simple_tree_layout(graph, root)
                    except:
                        self.metrics_output.append("Using spring layout as fallback for tree")
                        pos = nx.spring_layout(graph, seed=42)
                elif layout == 'radial':
                    pos = self._radial_layout(graph)
                elif layout == 'stress':
                    try:
                        pos = nx.kamada_kawai_layout(graph)
                    except:
                        self.metrics_output.append("Using spring layout as fallback for stress")
                        pos = nx.spring_layout(graph, seed=42)
                elif layout == 'multipartite':
                    pos = self._multipartite_layout(graph)
                else:
                    k_value = (1.0 / np.sqrt(graph.number_of_nodes()) + self.scale_factor * 0.1)
                    pos = nx.spring_layout(graph, k=k_value, iterations=200, seed=42)
            except Exception as e:
                self.metrics_output.append(f"Error computing {layout} layout: {str(e)}")
                self.metrics_output.append("Using spring layout as fallback")
                k_value = (1.0 / np.sqrt(graph.number_of_nodes()) + self.scale_factor * 0.1)
                pos = nx.spring_layout(graph, k=k_value, iterations=200, seed=42)

            # Apply scale factor to positions consistently across all layouts
            pos = {node: coord * self.scale_factor for node, coord in pos.items()}
            nx.set_node_attributes(graph, pos, 'pos')

        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            x_pad = 0.05 * x_range
            y_pad = 0.05 * y_range
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        size_attr = self.node_size_dropdown.currentText()
        if size_attr == 'uniform':
            node_size = 100 * self.scale_factor
        else:
            try:
                sizes = []
                for n in graph.nodes():
                    val = graph.nodes[n].get(size_attr, 1)
                    try:
                        sizes.append(float(val) * 100 * self.scale_factor)
                    except (ValueError, TypeError):
                        sizes.append(100 * self.scale_factor)
                node_size = sizes
            except Exception as e:
                self.metrics_output.append(f"Error processing size attribute: {e}")
                node_size = 100 * self.scale_factor

        color_attr = self.node_color_dropdown.currentText()
        if color_attr == 'uniform':
            node_color = self.node_color
        else:
            try:
                colors = [graph.nodes[n].get(color_attr, 0) for n in graph.nodes()]
                if any(isinstance(c, str) for c in colors):
                    unique_vals = list(set(colors))
                    color_map = {v: i for i, v in enumerate(unique_vals)}
                    colors = [color_map[c] for c in colors]
                node_color = colors
            except Exception as e:
                self.metrics_output.append(f"Error processing color attribute: {e}")
                node_color = self.node_color

        if (color_attr == 'uniform' and list(graph.nodes()) and
                'community' in graph.nodes[list(graph.nodes())[0]]):
            communities = [graph.nodes[n].get('community', 0) for n in graph.nodes()]
            node_color = communities

        edge_widths = []
        for u, v in graph.edges():
            try:
                weight = graph.get_edge_data(u, v, {}).get('weight', 1.0)
                edge_widths.append(float(weight) * 1 * self.scale_factor)
            except (KeyError, ValueError, TypeError):
                edge_widths.append(1.0 * self.scale_factor)

        # Draw edges
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5,
                              edge_color=self.edge_color,
                              width=edge_widths)

        # Draw nodes with interactivity
        nodes = nx.draw_networkx_nodes(graph, pos, ax=ax,
                                      node_size=node_size,
                                      node_color=node_color,
                                      node_shape=self.node_shape_dropdown.currentText(),
                                      cmap=plt.colormaps['tab20'] if isinstance(node_color, list) else None,
                                      alpha=0.8)
        nodes.set_picker(True)  # Enable picking for nodes

        if self.show_labels_checkbox.isChecked():
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=max(8 * self.scale_factor, 6))

        if list(graph.nodes()) and 'community' in graph.nodes[list(graph.nodes())[0]]:
            communities = [graph.nodes[n]['community'] for n in graph.nodes()]
            unique_communities = sorted(set(communities))
            community_counts = {comm: communities.count(comm) for comm in unique_communities}
            legend_elements = []
            cmap = plt.colormaps['tab20']
            for i, comm in enumerate(unique_communities):
                color = cmap(i / len(unique_communities))
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                              label=f'Community {comm} ({community_counts[comm]} nodes)',
                              markerfacecolor=color, markersize=10 * self.scale_factor)
                )
            plt.legend(handles=legend_elements, title='Detected Communities',
                      bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

        ax.axis('off')
        algo = self.community_dropdown.currentText() if self.community_results else "Louvain"
        ax.set_title(f"Network Visualization - {layout.capitalize()} Layout | Communities ({algo})")

        # Connect event handlers
        self.event_cids.append(self.figure.canvas.mpl_connect('pick_event', self.on_pick))
        self.event_cids.append(self.figure.canvas.mpl_connect('button_press_event', self.on_press))
        self.event_cids.append(self.figure.canvas.mpl_connect('button_release_event', self.on_release))
        self.event_cids.append(self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion))

        self.selected_node = None
        self.dragging = False
        self.canvas.draw()

    def on_pick(self, event):
        """Handle node selection on click"""
        artist = event.artist
        if isinstance(artist, matplotlib.collections.PathCollection):  # Check if it's a node collection
            ind = event.ind[0]
            nodes = list((self.G if self.filtered_graph is None else self.filtered_graph).nodes())
            self.selected_node = nodes[ind]
            self.metrics_output.append(f"Selected node: {self.selected_node}")
            self.canvas.draw()

    def on_press(self, event):
        """Start dragging if a node is selected"""
        if self.selected_node and event.inaxes:
            x, y = event.xdata, event.ydata
            pos = nx.get_node_attributes(self.G if self.filtered_graph is None else self.filtered_graph, 'pos')
            if self.selected_node in pos:
                self.drag_start = np.array([x, y])
                self.node_pos = pos[self.selected_node]
                self.dragging = True

    def on_motion(self, event):
        """Drag the selected node"""
        if self.dragging and event.inaxes and event.xdata and event.ydata:
            dx = event.xdata - self.drag_start[0]
            dy = event.ydata - self.drag_start[1]
            new_pos = self.node_pos + np.array([dx, dy])
            pos = nx.get_node_attributes(self.G if self.filtered_graph is None else self.filtered_graph, 'pos')
            pos[self.selected_node] = new_pos
            nx.set_node_attributes(self.G if self.filtered_graph is None else self.filtered_graph, pos, 'pos')
            self.draw_graph()

    def on_release(self, event):
        """Stop dragging"""
        self.dragging = False
        self.selected_node = None

    def _simple_tree_layout(self, G, root):
        """Simple tree layout using BFS"""
        pos = {}
        visited = set([root])
        pos[root] = np.array([0, 0])
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
                pos[node][0] = i - width/2
        return pos

    def _radial_layout(self, G):
        """Radial layout with nodes on concentric circles"""
        if not G.nodes():
            return {}
        try:
            if all('community' in G.nodes[n] for n in G.nodes()):
                groups = {n: G.nodes[n]['community'] for n in G.nodes()}
            else:
                groups = {n: 0 for n in G.nodes()}  # Default to a single group
        except KeyError:
            groups = {n: 0 for n in G.nodes()}  # Default to a single group if attributes are missing
        unique_groups = sorted(set(groups.values()))
        num_groups = len(unique_groups)
        pos = {}
        for i, group in enumerate(unique_groups):
            nodes_in_group = [n for n in G.nodes() if groups[n] == group]
            if not nodes_in_group:
                continue
            radius = 0.1 + 0.8 * (i / max(1, num_groups-1))
            theta = np.linspace(0, 2*np.pi, len(nodes_in_group) + 1, endpoint=False)
            for j, node in enumerate(nodes_in_group):
                pos[node] = np.array([radius * np.cos(theta[j]), radius * np.sin(theta[j])])
        return pos

    def _multipartite_layout(self, G):
        """Multipartite layout with layered node positioning"""
        if not G.nodes():
            return {}
        try:
            if all('layer' in G.nodes[n] for n in G.nodes()):
                layers = {n: G.nodes[n]['layer'] for n in G.nodes()}
            elif all('community' in G.nodes[n] for n in G.nodes()):
                layers = {n: G.nodes[n]['community'] for n in G.nodes()}
            else:
                layers = {n: 0 for n in G.nodes()}  # Default to a single layer
        except KeyError:
            layers = {n: 0 for n in G.nodes()}  # Default to a single layer if attributes are missing
        unique_layers = sorted(set(layers.values()))
        num_layers = len(unique_layers)
        pos = {}
        for i, layer in enumerate(unique_layers):
            nodes_in_layer = [n for n in G.nodes() if layers[n] == layer]
            if not nodes_in_layer:
                continue
            y = 1 - (i / max(1, num_layers-1))
            x_positions = np.linspace(0, 1, len(nodes_in_layer)) if len(nodes_in_layer) > 1 else [0.5]
            for j, node in enumerate(nodes_in_layer):
                pos[node] = np.array([x_positions[j], y])
        return pos

    def show_metrics(self):
        """Display network metrics"""
        if self.G is None:
            self.metrics_output.append("Please load a graph first")
            return
        self.metrics_output.clear()
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
            "Clustering Coefficient": clustering_coeffs,
            "Transitivity": nx.transitivity(self.G),
            "Diameter": self._calculate_diameter(self.G),
            "Average Path Length": self._calculate_avg_path_length(self.G)
        }
        output = []
        output.append("{:<30} {:<15}".format('Metric', 'Value'))
        output.append("-" * 45)
        for name, value in metrics.items():
            output.append("{:<30} {:<15}".format(name, str(value)))
        degrees_list = list(degrees_dict.values())
        output.append("\n{:<30} {:<15}".format('Degree Distribution', ''))
        output.append("{:<30} {:<15.2f}".format('Min degree', min(degrees_list)))
        output.append("{:<30} {:<15.2f}".format('Max degree', max(degrees_list)))
        output.append("{:<30} {:<15.2f}".format('Median degree', np.median(degrees_list)))
        output.append("{:<30} {:<15.2f}".format('Degree std.dev', np.std(degrees_list)))
        output.append("\n{:<10} {:<15} {:<15} {:<15}".format(
            'Node', 'Degree' if not self.directed else 'Total Degree',
            'In-Degree' if self.directed else '', 'Out-Degree' if self.directed else ''
        ))
        output.append("-" * 50)
        for node in self.G.nodes():
            degree = degrees_dict.get(node, 0)
            in_deg = in_degrees.get(node, 0) if self.directed else ""
            out_deg = out_degrees.get(node, 0) if self.directed else ""
            output.append("{:<10} {:<15} {:<15} {:<15}".format(str(node), str(degree), str(in_deg), str(out_deg)))
        if self.directed:
            in_degrees_list = list(in_degrees.values())
            out_degrees_list = list(out_degrees.values())
            output.append("\n{:<30} {:<15}".format('In-Degree Distribution', ''))
            output.append("{:<30} {:<15.2f}".format('Average in-degree', np.mean(in_degrees_list)))
            output.append("{:<30} {:<15.2f}".format('Min in-degree', min(in_degrees_list)))
            output.append("{:<30} {:<15.2f}".format('Max in-degree', max(in_degrees_list)))
            output.append("\n{:<30} {:<15}".format('Out-Degree Distribution', ''))
            output.append("{:<30} {:<15.2f}".format('Average out-degree', np.mean(out_degrees_list)))
            output.append("{:<30} {:<15.2f}".format('Min out-degree', min(out_degrees_list)))
            output.append("{:<30} {:<15.2f}".format('Max out-degree', max(out_degrees_list)))
        if clustering_coeffs:
            coeff_values = list(clustering_coeffs.values())
            num_nodes = len(coeff_values)
            output.append("\n{:<30} {:<15}".format('Clustering Coefficient', ''))
            output.append("{:<30} {:<15.2f}".format('Average coefficient', avg_clustering))
            output.append("{:<30} {:<15.2f}".format('Min coefficient', min(coeff_values)))
            output.append("{:<30} {:<15.2f}".format('Max coefficient', max(coeff_values)))
            output.append("{:<30} {:<15.2f}".format('Median coefficient', np.median(coeff_values)))
            output.append("{:<30} {:<15.2f}".format('Coefficient std.dev', np.std(coeff_values)))
            if num_nodes <= 50:
                output.append("\nIndividual Clustering Coefficients:")
                for node, coeff in list(clustering_coeffs.items())[:50]:
                    output.append(f"Node {node}: {coeff:.4f}")
            elif num_nodes <= 200:
                sample_nodes = list(clustering_coeffs.keys())[:5]
                output.append("\nSample Clustering Coefficients:")
                for node in sample_nodes:
                    output.append(f"Node {node}: {clustering_coeffs[node]:.4f}")
                output.append(f"\n... and {num_nodes-5} more nodes")
            else:
                output.append("\n(Network too large to display individual coefficients)")
        self.metrics_output.setPlainText("\n".join(output))
        self.metrics_output.moveCursor(QTextCursor.Start)

    def _calculate_diameter(self, G):
        """Calculate diameter handling disconnected graphs"""
        if self.directed:
            if nx.is_weakly_connected(G):
                try:
                    return nx.diameter(G.to_undirected(as_view=True))
                except nx.NetworkXError:
                    return "Disconnected (no valid diameter)"
            else:
                components = list(nx.weakly_connected_components(G))
                largest_comp = max(components, key=len, default=set())
                if len(largest_comp) > 1:
                    subgraph = G.subgraph(largest_comp).to_undirected(as_view=True)
                    try:
                        return f"Disconnected (largest component: {nx.diameter(subgraph)})"
                    except nx.NetworkXError:
                        return "Disconnected (no valid diameter in largest component)"
                return "Disconnected (no valid components)"
        else:
            if nx.is_connected(G):
                return nx.diameter(G)
            else:
                components = list(nx.connected_components(G))
                largest_comp = max(components, key=len, default=set())
                if len(largest_comp) > 1:
                    return f"Disconnected (largest component: {nx.diameter(G.subgraph(largest_comp))})"
                return "Disconnected (no valid components)"

    def _calculate_avg_path_length(self, G):
        """Calculate average path length handling disconnected graphs"""
        if self.directed:
            if nx.is_weakly_connected(G):
                try:
                    return nx.average_shortest_path_length(G.to_undirected(as_view=True))
                except nx.NetworkXError:
                    return "Disconnected (no valid paths)"
            else:
                components = list(nx.weakly_connected_components(G))
                avg_lengths = []
                for comp in components:
                    subgraph = G.subgraph(comp).to_undirected(as_view=True)
                    if len(comp) > 1:
                        try:
                            avg_lengths.append(nx.average_shortest_path_length(subgraph))
                        except nx.NetworkXError:
                            continue
                if avg_lengths:
                    return f"Disconnected (avg: {np.mean(avg_lengths):.2f})"
                return "Disconnected (no valid components)"
        else:
            if nx.is_connected(G):
                return nx.average_shortest_path_length(G)
            else:
                components = list(nx.connected_components(G))
                avg_lengths = []
                for comp in components:
                    subgraph = G.subgraph(comp)
                    if len(comp) > 1:
                        try:
                            avg_lengths.append(nx.average_shortest_path_length(subgraph))
                        except nx.NetworkXError:
                            continue
                if avg_lengths:
                    return f"Disconnected (avg: {np.mean(avg_lengths):.2f})"
                return "Disconnected (no valid components)"

    def _check_resources(self):
        """Check available system resources (CPU and memory)"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            # Define thresholds for resource usage
            cpu_threshold = 90  # CPU usage above 90% is considered high
            memory_threshold = 90  # Memory usage above 90% is considered high
            if cpu_percent > cpu_threshold or memory_percent > memory_threshold:
                return False, f"High resource usage detected: CPU {cpu_percent}%, Memory {memory_percent}%"
            return True, ""
        except Exception as e:
            return False, f"Error checking resources: {str(e)}"

    def _calculate_conductance(self, G, communities):
        """Calculate conductance for communities, avoiding double-counting in undirected graphs"""
        if len(communities) <= 1 or not communities:
            return 0.0
        total_conductance = 0
        for community in communities:
            if not community:
                continue
            community_set = set(community)
            internal_edges = 0
            external_edges = 0
            # Track edges to avoid double-counting in undirected graphs
            counted_edges = set()
            for node in community:
                for neighbor in G.neighbors(node):
                    edge = tuple(sorted([node, neighbor])) if not self.directed else (node, neighbor)
                    if edge in counted_edges:
                        continue
                    counted_edges.add(edge)
                    if neighbor in community_set:
                        internal_edges += 1
                    else:
                        external_edges += 1
            # Adjust internal edges for undirected graphs (each edge was counted once per direction)
            if not self.directed:
                internal_edges = internal_edges / 2
            if internal_edges + external_edges == 0:
                community_conductance = 0
            else:
                community_conductance = external_edges / (internal_edges + external_edges)
            total_conductance += community_conductance
        return total_conductance / len(communities)

    def _calculate_silhouette(self, G, partition):
        """Calculate silhouette coefficient with sampling for large graphs"""
        if len(set(partition.values())) <= 1 or not partition:
            return 0.0
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        if len(communities) <= 1:
            return 0
        nodes = list(G.nodes())
        n = len(nodes)
        # Sample for large graphs
        if n > 200:
            sample_size = min(200, n)
            sample_nodes = np.random.choice(nodes, sample_size, replace=False)
        else:
            sample_nodes = nodes
        total_silhouette = 0
        count = 0
        for node in sample_nodes:
            if node not in partition:
                continue
            community_id = partition[node]
            own_community = communities[community_id]
            if len(own_community) <= 1:
                continue
            own_distances = []
            for other_node in own_community:
                if other_node != node:
                    try:
                        dist = 1 if G.has_edge(node, other_node) else nx.shortest_path_length(G, node, other_node)
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
                        dist = 1 if G.has_edge(node, other_node) else nx.shortest_path_length(G, node, other_node)
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
        return total_silhouette / count if count > 0 else 0.0

    def _calculate_adjusted_rand_index(self, labels1, labels2):
        """Calculate adjusted Rand index using sklearn for efficiency"""
        if not labels1 or not labels2 or len(labels1) != len(labels2):
            return 0
        return adjusted_rand_score(labels1, labels2)

    def _calculate_variation_of_information(self, labels1, labels2):
        """Calculate variation of information"""
        if not labels1 or not labels2 or len(labels1) != len(labels2):
            return float('inf')
        labels1 = np.array(labels1)
        labels2 = np.array(labels2)
        n = len(labels1)

        # Calculate entropy for labels1
        entropy1 = 0.0
        for label in set(labels1):
            p = np.sum(labels1 == label) / n
            if p > 0:
                entropy1 -= p * np.log2(p)

        # Calculate entropy for labels2
        entropy2 = 0.0
        for label in set(labels2):
            p = np.sum(labels2 == label) / n
            if p > 0:
                entropy2 -= p * np.log2(p)

        # Calculate mutual information
        mi = 0.0
        for label1 in set(labels1):
            for label2 in set(labels2):
                joint_p = np.sum((labels1 == label1) & (labels2 == label2)) / n
                if joint_p > 0:
                    p1 = np.sum(labels1 == label1) / n
                    p2 = np.sum(labels2 == label2) / n
                    mi += joint_p * np.log2(joint_p / (p1 * p2))

        return entropy1 + entropy2 - 2 * mi

    def _directed_modularity(self, partition, G):
        """Calculate modularity for directed graphs"""
        if not G.is_directed():
            return community_louvain.modularity(partition, G)
        m = G.number_of_edges()
        if m == 0:
            return 0.0
        modularity = 0.0
        degrees_out = dict(G.out_degree())
        degrees_in = dict(G.in_degree())
        communities = set(partition.values())
        for c in communities:
            nodes_in_c = [n for n, comm in partition.items() if comm == c]
            if not nodes_in_c:
                continue
            e_cc = sum(1 for u, v in G.edges() if partition[u] == c and partition[v] == c)
            out_deg_c = sum(degrees_out.get(n, 0) for n in nodes_in_c)
            in_deg_c = sum(degrees_in.get(n, 0) for n in nodes_in_c)
            if m > 0:
                modularity += (e_cc / m) - (out_deg_c * in_deg_c / (m * m))
        return modularity

    def link_analysis(self):
        """Perform link analysis and calculate centrality measures"""
        if self.G is None:
            self.link_output.append("Please load a graph first")
            return
        self.link_output.clear()
        self.centrality_measures = {}
        self.centrality_ranges = {}
        progress = QProgressDialog("Calculating centrality measures...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Link Analysis")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            # Degree centrality
            progress.setValue(10)
            QApplication.processEvents()
            degree_centrality = nx.degree_centrality(self.G)
            self.centrality_measures['Degree'] = degree_centrality
            degree_values = list(degree_centrality.values())
            self.centrality_ranges['Degree'] = {
                'min': min(degree_values, default=0),
                'max': max(degree_values, default=0)
            }

            # Betweenness centrality
            progress.setValue(30)
            QApplication.processEvents()
            betweenness_centrality = nx.betweenness_centrality(self.G, normalized=True)
            self.centrality_measures['Betweenness'] = betweenness_centrality
            betweenness_values = list(betweenness_centrality.values())
            self.centrality_ranges['Betweenness'] = {
                'min': min(betweenness_values, default=0),
                'max': max(betweenness_values, default=0)
            }

            # Closeness centrality
            progress.setValue(50)
            QApplication.processEvents()
            closeness_centrality = nx.closeness_centrality(self.G)
            self.centrality_measures['Closeness'] = closeness_centrality
            closeness_values = list(closeness_centrality.values())
            self.centrality_ranges['Closeness'] = {
                'min': min(closeness_values, default=0),
                'max': max(closeness_values, default=0)
            }

            # Eigenvector centrality
            progress.setValue(70)
            QApplication.processEvents()
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.G, max_iter=1000, tol=1e-06)
            except nx.PowerIterationFailedConvergence:
                self.link_output.append("Warning: Eigenvector centrality did not converge, using approximation")
                eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G)
            self.centrality_measures['Eigenvector'] = eigenvector_centrality
            eigenvector_values = list(eigenvector_centrality.values())
            self.centrality_ranges['Eigenvector'] = {
                'min': min(eigenvector_values, default=0),
                'max': max(eigenvector_values, default=0)
            }

            # PageRank
            progress.setValue(90)
            QApplication.processEvents()
            pagerank = nx.pagerank(self.G, alpha=0.85)
            self.centrality_measures['PageRank'] = pagerank
            pagerank_values = list(pagerank.values())
            self.centrality_ranges['PageRank'] = {
                'min': min(pagerank_values, default=0),
                'max': max(pagerank_values, default=0)
            }

            progress.setValue(100)
            progress.close()

            # Output centrality measures
            self.link_output.append("Centrality Measures\n")
            self.link_output.append("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
                'Node', 'Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'PageRank'
            ))
            self.link_output.append("-" * 95)
            for node in self.G.nodes():
                deg = self.centrality_measures['Degree'].get(node, 0)
                bet = self.centrality_measures['Betweenness'].get(node, 0)
                clo = self.centrality_measures['Closeness'].get(node, 0)
                eig = self.centrality_measures['Eigenvector'].get(node, 0)
                pr = self.centrality_measures['PageRank'].get(node, 0)
                self.link_output.append(
                    "{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                        str(node), deg, bet, clo, eig, pr
                    )
                )

            # Summary statistics
            self.link_output.append("\nSummary Statistics\n")
            for measure, values in self.centrality_measures.items():
                vals = list(values.values())
                if vals:
                    self.link_output.append(f"\n{measure}:")
                    self.link_output.append(f"Min: {min(vals):.4f}")
                    self.link_output.append(f"Max: {max(vals):.4f}")
                    self.link_output.append(f"Mean: {np.mean(vals):.4f}")
                    self.link_output.append(f"Median: {np.median(vals):.4f}")
                    self.link_output.append(f"Std Dev: {np.std(vals):.4f}")

            self.link_output.moveCursor(QTextCursor.Start)

        except Exception as e:
            progress.close()
            self.link_output.append(f"Error during link analysis: {str(e)}")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    tool = EnhancedNetworkAnalysisTool()
    sys.exit(app.exec_())