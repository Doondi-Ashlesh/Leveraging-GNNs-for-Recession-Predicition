# Recession Prediction using Graph Neural Networks (GNNs)

This repository presents a novel framework that leverages **Graph Neural Networks (GNNs)**—specifically **GCN** and **TGNN** architectures—to predict U.S. economic recessions. The system integrates graph-theoretic modeling of macroeconomic indicators with deep learning, offering early-warning insights into structural economic shifts.

> **Project Report**: [`Report.pdf`](./Report.pdf)  
> **Live Inference**: Gradio App via [`Interface.py`](./Interface.py)

---

## Description

Traditional econometric and time-series models often miss complex nonlinear dependencies between macroeconomic indicators. In this work, we:

- Construct dynamic graphs using Pearson correlation among 22 financial indicators.
- Train GNN models (GCN, TGNN) to learn both spatial and temporal interdependencies.
- Address class imbalance using SMOTE and Focal Loss.
- Deploy a user-friendly interface for real-time inference using a trained TGNN.

---

## Dataset

- **Source**: [Kaggle – Financial Indicators of US Recession](https://www.kaggle.com/datasets/rohanrao/financial-indicators-of-us-recession)
- **Indicators**: 27 (e.g., GDP, Unemployment Rate, CPI, M2, Federal Funds Rate)
- **Labels**: Monthly recession labels from NBER (1985–2023)
- **Preprocessing**:
  - z-score normalization
  - SMOTE-based balancing
  - Pearson correlation-based graph construction

---

## Models

### Graph Convolutional Network (GCN)

- Static graph  
- 4-layer GCN with dropout and mean pooling  
- Focal Loss for training

### Temporal Graph Neural Network (TGNN)

- Dynamic graphs over time  
- GCN + GRU architecture  
- Achieved best performance

---

## Demo: Gradio Interface

We provide a Gradio-based GUI (`Interface.py`) to predict recession probabilities from new data.

---

### Execution Instructions

---

### Required Libraries and Installation Commands

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install torch
pip install torch-geometric
pip install networkx
pip install imbalanced-learn
pip install gradio
pip install joblib
pip install tqdm  # optional for progress bars
```

> **Note**: PyTorch Geometric requires compatibility with your PyTorch and CUDA versions. Use their official install selector.

---

### Code Execution

#### Quick Demo Interface

This is a simplified script to test predictions using our trained TGNN model via a Gradio interface. You can simply run `Interface.py` to see the results (predictions) instead of executing the whole source code.

---

#### Steps to Run the `Interface.py` File

1. **Files Required in the Same Folder**

Make sure the following files are all in the same folder with `Interface.py`. The folder already contains the trained and saved files for the TGNN model. If testing a different model, define its architecture in the code and ensure all associated files are available:

- `Interface.py`
- `TGNN_SMOTE_focal_40e.pth` – Trained model file
- `scaler_tgnn.pkl` – StandardScaler used during training
- `feature_list.pkl` – Feature name list
- `graph_edges.pt` – Contains edge_index and edge_weight for the model

2. **Activate Environment & Install Minimal Dependencies**

```bash
pip install torch numpy gradio scikit-learn joblib
```

3. **Run Interface**

Open `Interface.py` and execute the code.

4. **Access Interface**

Click on the localhost link in the output terminal or open browser at:

```
http://127.0.0.1:7860
```

5. **Input Features**

Enter a comma-separated string of feature values (excluding date and recession label). Sample inputs are provided in `sample_inputs_for_interface.txt`.

6. **Output**

- Prediction: Recession / No Recession
- Probabilities for each class

---

#### Steps to Run the Whole Original Source Code

1. **Ensure Input Dataset is Available**

Ensure the following file is in your working directory:

- `filled_temp_dataset.csv`  
  *(Renamed to `Custom_dataset` in the `dataset` folder — update the dataset path accordingly)*

2. **Launch Jupyter Notebook**

Open `GNN_pipeline.ipynb` and run all cells sequentially.

> The notebook performs:
> - Preprocessing of the dataset  
> - Graph construction and modeling  
> - Training using GCN/TGNN  
> - Evaluation and metric plotting  
> - Interface block for prediction

If configured, outputs like confusion matrices, probabilities, and visualizations will appear in the notebook. You may optionally save models and required files (`scaler.pkl`, `feature_list.pkl`, `graph_edges.pt`) for interface testing (modifications needed in interface code).

---

THE END
