# TriangleMesh: Deep Networks for 3D Mesh Classification

This repository contains the official implementation and hybrid architecture (MeshCNN + DGCNN) described in the research paper: **"Decoding 3D Geometry: Deep Networks for Mesh Classification"**  

---

## 🚀 Research & Field Contributions

This project addresses the challenges of irregular 3D mesh structures in deep learning. Our work provides three primary contributions to the AI community:
1. **Hybrid Architecture:** A novel model fusing edge-based (MeshCNN) and vertex-based (DGCNN) features for robust geometric understanding.
2. **Standardized Benchmarking:** A systematic hyperparameter analysis providing a foundation for future mesh-native neural network development.
3. **Cleaned ModelNet Dataset Release:** The identification and resolution of significant geometric instabilities in standard industry benchmarks.

### 📦 Cleaned ModelNet Dataset Release
As detailed in **Section III-C** of our paper, the original ModelNet benchmarks contain significant geometric and topological artifacts (e.g., zero-area faces and unused vertices) that cause runtime instabilities and heap errors in mesh-native networks. 

To facilitate stable, reproducible research for the AI community, we have released a cleaned version of the dataset with the following fixes:
*   **Geometric Validation:** Removal of degenerate/collinear faces.
*   **Topology Correction:** Removal of unused vertices and re-indexing of mesh connectivity.
*   **Data Integrity:** Standardized Object File Format (OFF) structures for seamless training.

**Download the Cleaned Dataset here:** [Cleaned ModelNet (3D Mesh)](https://drive.google.com/file/d/1UO8qF56AvrjeCXt4-nRs8rv0Ukmr2Rlg/view?usp=sharing)

---

## 🛠 Usage

### 1. Data Preparation
*   Run the provided script: `get_data.sh`
*   **Alternatively:** Manually download and extract the SHREC_16 data from [this source](https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz) into the `data/` folder.
*   To use the **Cleaned ModelNet** dataset, extract it into `data/modelnet_cleaned/`.

### 2. Configuration
Create a configuration YAML file using the template provided in `configs/`. 
*   All necessary instructions for hyperparameter fields (feature channels, pooling edges, learning rates) are contained within the sample YAML.
*   Note: While default values are provided, all fields in the YAML must be filled to initialize the model.

### 3. Training & Evaluation
To train the model:
```bash
python main.py train CONFIG_YAML_PATH
```

To evaluate the model on a test set:
```bash
python main.py test CONFIG_YAML_PATH
```

---

## 📚 Citation
If you use this codebase or the cleaned ModelNet dataset in your research, please cite our work:

```bibtex
@article{syed2022decoding,
  title={Decoding 3D Geometry: Deep Networks for Mesh Classification},
  author={Narayanan, A. S. and Gupta, E. and Syed, J. and Carmon, M.},
  journal={Georgia Institute of Technology},
  year={2022}
}
```
