# Dynamic Scenes: 3D Point Cloud Change Detection, Alignment, and Spatial Partitioning

This repository contains a high-performance Python implementation of an end-to-end pipeline for 3D point cloud analysis, change detection, and geometric object classification. Developed as part of the "3D Computational Geometry and Vision" course in the Department of Electrical and Computer Engineering at the University of Patras, this project leverages spatial data structures and robust algorithms to analyze landscape-level differences between multi-epoch aerial LIDAR scans (specifically 2016 and 2020 datasets).

The pipeline incorporates advanced ground filtering, combined-cloud Iterative Closest Point (ICP) registration, recursive spatial partitions (KD-Trees, Octrees, and Quadtrees), geometric shape signature matching, and voxel-grid-accelerated Euclidean clustering for semantic classification.

---

## Core Pipeline and Methodology

### 1. Robust Ground Filtering
To isolate above-ground structures (such as buildings, vegetation, and vehicles), the system implements a grid-based Local Minimum Filter:
* **Cellular Discretization**: The horizontal (X-Y) plane is discretized into a regular grid of cells with user-defined spatial resolution.
* **Elevation Analysis**: For each grid cell, the local minimum elevation (Z) is computed, providing an accurate, localized ground baseline.
* **Adaptive Thresholding**: Points lying within a vertical clearance threshold above the cell's local minimum are flagged as ground and thinned. This approach handles steep slopes and uneven terrain far more reliably than global elevation percentiles.

### 2. Registration and Alignment
Multi-epoch point clouds can exhibit global translation and rotation offsets due to sensor coordinate differences or coordinate system discrepancies. The system resolves this via a two-stage registration process:
* **Combined-Cloud PCA Warm-Start**: Principal Component Analysis (PCA) is performed on the combined set of points from both epochs. Points are rotated into a shared coordinate frame. Constraining the rotation strictly to the vertical (height) axis ensures that ground orientations remain consistent and prevents the eigenvector orientation flip common in independent PCA.
* **Iterative Closest Point (ICP) Refinement**: Point-to-point SVD (Singular Value Decomposition) alignment is applied to register the source cloud (2016) onto the target reference cloud (2020). High-speed nearest-neighbor queries are executed via vectorised, batch-processed NumPy routines.

### 3. Vectorised Point-Level Change Detection
Once aligned, points from the 2016 and 2020 epochs are matched using distance-based thresholds:
* **Static Points**: Points in the 2020 epoch that lie within a threshold distance of a point in the 2016 epoch.
* **Added Points**: Points present in the 2020 epoch but missing from the 2016 epoch (representing new structures or objects).
* **Removed Points**: Points present in the 2016 epoch but missing from the 2020 epoch (representing demolished structures or displaced objects).

### 4. Hierarchical Spatial Partitioning
The project implements three custom recursive spatial trees in the `src/` directory to structure and query the datasets:
* **K-Dimensional Tree (KD-Tree)**: Built recursively by splitting points at the median along alternating coordinate axes. Supports exact/approximate Nearest Neighbor search, exact K-Nearest Neighbors (k-NN) via max-heaps, and Spherical Range Search.
* **Octree**: A 3D spatial partitioning tree where nodes with counts exceeding the maximum capacity are split recursively into eight child octants. Used for volumetric change detection.
* **Quadtree**: A 2D spatial partitioning tree operating on the horizontal plane. Nodes are split recursively into four quadrant children. Used for top-down spatial change queries.

### 5. Advanced Geometric Shape Matching
Instead of simple point-count comparisons, the system evaluates structural changes using advanced geometric criteria:
* **PCA Eigenvalue Signature**: For each leaf node in the spatial trees, the local covariance matrix is computed. The three sorted eigenvalues are normalized to sum to 1. This three-element descriptor compactly represents whether the local shape is linear, planar, or volumetric. Significant differences in eigenvalue signatures indicate structural morphology changes.
* **Directed Hausdorff Distance**: The directed Hausdorff distance between corresponding leaf nodes in two epochs is approximated using batch vectorised pairwise distance operations. Substantial shifts in physical locations flag a node as changed.

### 6. Voxel-Grid Euclidean Clustering and Classification
To segment individual dynamic objects and identify their types, the pipeline processes the point sets using:
* **Accelerated Euclidean Clustering**: A custom DBSCAN-like algorithm. To avoid the O(N^2) complexity of pairwise distances, a voxel-grid spatial index maps points to 3D bins. Radius queries only inspect adjacent 27 voxels, reducing clustering time to near-linear complexity.
* **Geometric Bounding Box Heuristics**: Extracted clusters are classified into four distinct categories based on their 3D extents:
  * **Building**: High horizontal footprint area and moderate-to-high vertical height.
  * **Tree**: Moderate-to-high vertical height combined with a compact horizontal footprint area.
  * **Pole/Mast**: Extremely narrow horizontal footprint and distinct vertical height.
  * **Object**: General structures and clutter that do not match the above signatures.

---

## Project Structure

```
├── Data/
│   ├── 2016/                         # 2016 LIDAR datasets (.laz format)
│   └── 2020/                         # 2020 LIDAR datasets (.laz format)
├── documentation/                    # Documentation, HTML assets, and user guide
├── src/
│   ├── kdnode.py                     # Recursive KD-Tree implementation and spatial queries
│   ├── octnode.py                    # Octree implementation for volumetric subdivision
│   ├── quadnode.py                   # Quadtree implementation for horizontal spatial tracking
│   └── readlazfiles.py               # Helper utility to read and parse .laz LIDAR files
├── vvrpywork/                        # 3D visualization and UI engine
├── dynamicScenes.py                  # Main execution script, visualization controller, and algorithms
├── README.md                         # Project documentation
└── Project_4_2026.pdf                # Course project guidelines and specifications
```

---

## Installation and Setup

### Prerequisites
Ensure you have Python 3.12 or newer installed.

### Dependencies
Install the required packages using pip:
```bash
pip install numpy laspy lazrs
```
Note: `lazrs` is required as a decompression backend to parse `.laz` format point clouds using `laspy`.

---

## Usage and Interactive Controls

To execute the project and launch the interactive 3D visualizer:
```bash
python dynamicScenes.py
```

Upon startup, the script performs the initial grid-based ground filtering, combined-cloud PCA projection, and SVD-based ICP alignment. Once the preprocessing phase is complete, a 3D rendering window will open.

You can switch between different analysis tasks and visualization modes using the keyboard:

* **Key 1 (or Numpad 1)**: Renders the dynamic scene classification. Static points are clustered and categorized into buildings (cyan), trees (green), poles (yellow), and objects (orange), while raw added (green) and removed (red) points are rendered directly.
* **Key 2 (or Numpad 2)**: Renders raw scene differences. Static points are colorized continuously by elevation (Y-axis), added points are colored red, and removed points are colored blue.
* **Key 3 (or Numpad 3)**: Renders volumetric change detection using Octrees. The visualizer overlays generalized transparent bounding boxes over the three-dimensional regions where changes were detected.
* **Key 4 (or Numpad 4)**: Renders horizontal change detection using Quadtrees. The visualizer draws vertical columns enclosing sections of spatial horizontal change.
