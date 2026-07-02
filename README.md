# Dynamic Scenes: 3D Point Cloud Change Detection, Alignment, and Semantic Classification

This repository contains a high-performance Python implementation of an end-to-end pipeline for 3D point cloud analysis, change detection, and geometric object classification. Developed as part of the "3D Computational Geometry and Vision" course in the Department of Electrical and Computer Engineering at the University of Patras, this project leverages spatial data structures, multi-threaded GUI runtimes, and robust computational geometry algorithms.

The pipeline is split into two primary operational modes:
1. **Dynamic PCD Sequence Player**: Animates continuous frame-by-frame point cloud sequences (e.g., from moving vehicle lidars) to run height-coloring and frame-to-frame change detection.
2. **Static LIDAR**: Analyzes multi-epoch scans (2016 and 2020 LIDAR epochs).

---

## Architecture and Design

To ensure optimal responsiveness and clean architecture, the repository has been fully modularized:

### 1. Decoupled Multi-threaded Execution (`workers.py`)
To prevent heavy CPU computations (file I/O, ego-motion matrix calculations, ground grid filtering, and DBSCAN clustering) from freezing the visualizer's rendering context, calculations are offloaded to an asynchronous background `FrameProcessingWorker` (`QThread`). The main thread remains dedicated to drawing and interacting with the scene at a smooth **60 FPS**, receiving processed datasets via thread-safe PyQt signals.

### 2. In-Memory Circular Buffering (`loaders.py`)
When playing dynamic sequences, reading large coordinate files from disk continuously introduces I/O latency. The `PCDSequenceLoader` maintains a size-bounded, FIFO cache (max size 30) in RAM. Previously read frames are served instantly, enabling smooth, stutter-free animation loops.

### 3. Voxel-Grid Spatial Partitioning DBSCAN (`pipeline.py`)
In order to segment structures in real-time, we implement a custom DBSCAN algorithm. Space is discretized into a 3D grid of size $\epsilon$ (voxels). This way, nearest-neighbor searches only evaluate points within the cell and its 26 immediate neighbor voxels, scaling clustering queries to $O(N)$ linear complexity.

### 4. Interactive PyQt Control Panel
Both visualizers feature an interactive side control panel, exposing:
* **Interactive Checkboxes**: Toggle the visibility of Buildings, Trees, Poles, and general Objects layers dynamically.
* **Single-Cluster Focus Selector**: A combo box that automatically populates with all clusters found in the current frame, allowing the user to select and isolate a single object in the 3D space.

---

## Directory Structure

```
├── Data/
│   ├── pcd/                          # Calibration CSV and PCD frame sequences
│   └── laz/                          # Aerial LIDAR datasets (.laz format)
├── documentation/                    # HTML guides and visualization documentation
├── src/
│   ├── geometry_utils.py             # Math helpers (SVD-based ICP, PCA, transform matrices, scaling)
│   ├── pipeline.py                   # Preprocessing (ground filter, difference vectors, DBSCAN)
│   ├── loaders.py                    # Data loader classes for PCD and LIDAR sequences
│   ├── workers.py                    # Asynchronous PyQt FrameProcessingWorker thread
│   ├── readpcdfiles.py               # Low-level ASCII/Binary PCD file parser
│   ├── readlazfiles.py               # laz/las parsing utility wrapper
│   ├── kdnode.py                     # Custom recursive KD-Tree implementation
│   ├── octnode.py                    # Custom recursive Octree implementation
│   └── quadnode.py                   # Custom recursive Quadtree implementation
├── vvrpywork/                        # 3D PyVistaqt window engine wrappers
├── dynamicScenes.py                  # CLI argparse router and DynamicSceneViewer class
└── Project_4_2026.pdf                # Course specification sheet
```

---

## Installation and Requirements

### Prerequisites
* **Python**: Version 3.10 to 3.12 is recommended.
* **Anaconda Environment**: Setting up a dedicated environment is highly recommended.

### Package Installation
Install dependencies via pip:
```bash
pip install numpy scipy laspy lazrs pyvista pyvistaqt PyQt5
```
> Note: `lazrs` is required to allow `laspy` to decompress and parse `.laz` datasets.

---

## Usage and CLI Routing

The main script (`dynamicScenes.py`) uses `argparse` to inspect arguments. It dynamically routes the input based on the number and format of the files specified:

### 1. Launch PCD Sequence Viewer (Default)
Runs sequence loading and animates sequential frames:
```bash
python dynamicScenes.py
```
*Alternatively, supply a custom calibration file:*
```bash
python dynamicScenes.py Data/pcd/calibration.csv
```

### 2. Launch Aerial LIDAR Change Viewer
Runs registration and compares custom multi-epoch topography (supply 2 files):
```bash
python dynamicScenes.py Data/laz/2016.laz Data/laz/2020.laz
```

---

## Interactive Controls Reference

### 1. PCD Dynamic Scene Controls
* **Key `1`**: Switch to **Raw Height Mode** (height-gradient colorization).
* **Key `2`**: Switch to **Change Detection Mode** (Green: points added, Red: points removed, Grey: static).
* **Key `3`**: Switch to **Semantic Classification Mode** (Cyan: buildings, Green: vegetation/trees, Yellow: poles, Orange: objects).
* **Key `SPACE`**: Toggle play/pause animation playback. *(Pressing SPACE in Classification mode automatically reverts back to Raw mode to maintain animation speed).*
* **Key `N` / `P`**: Step Forward / Backward by one frame.
* **Key `R`**: Reset sequence playback to Frame 1.

### 2. Aerial LIDAR Viewer Controls
* **Key `1`**: **KD-Tree Dynamic Objects**. Computes vectorized nearest-neighbor checks to find changes, and groups static points into classified clusters.
* **Key `2`**: **Raw Differences**. Static points are colored by elevation, added points are green, and removed points are red.
* **Key `3`**: **Octree Volumetric Subdivision**. Visualizes volumetric modifications using transparent general bounding box grids.
* **Key `4`**: **Quadtree Horizontal Subdivision**. Visualizes modifications using vertical projection columns.

*Use the checkboxes and drop-down menu on the left side of the window to filter layers and isolate specific clusters in both modes.*
