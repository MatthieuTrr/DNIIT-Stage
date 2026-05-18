# Software Defect Prediction via DH-CNN

This project implements the DH-CNN (Deep Hierarchical Convolutional Neural Network) model for software defect prediction, as described in the research paper "Software Defect Prediction Based on Deep Representation Learning of Source Code From Contextual Syntax and Semantic Graph".

## Requirements

* **Python :** Version `3.9.6`
* **Material (Optional but recommended) :** Intel Core i7 with NVIDIA GeForce MX250 CUDA 11.2

## Installation

1. Clone the repository to your local machine.
2. Install the required dependencies via `pip` :

in dev environment :

``` bash
pip install -r requirements-dev.txt
```

in prod environment :

```bash
pip install -r requirements.txt
```

## Run tests
To ensure the integrity of the codebase, run the tests using `pytest` :

```bash
pytest tests/
```

Or without logs :

```bash
pytest -p no:cacheprovider tests/
```
## Project Architecture and Usage

The defect prediction process is divided into three main phases 
The scripts must be executed in this order :

### Step 1 : Parsing Source Code
This step analyzes the Java source files of the dataset (PROMISE dataset) to extract three representations of the code :
* **AST (Abstract Syntax Tree) :** Extraction of the contextual syntax using `javalang`.
* **CFG (Control Flow Graph) & DDG (Data Dependence Graph) :** Extraction of the graphical semantics using `networkx` and `NLTK`.

### Step 2 : Mapping Vectors (Vectorisation)
The neural networks require fixed-size numerical inputs.
* Run the embedding script to convert the AST nodes into numerical vectors using **Word2vec** (with `Gensim`).
* Run the embedding script to encode the CFG and DDG graphs into numerical vectors using **Node2vec**.

### Step 3 : Prediction Model (Training and Prediction DH-CNN)
The DH-CNN model uses `TensorFlow 2.5.0` and `Keras 2.5.0`.
* **Training :** The model learns simultaneously the syntactic (AST level) and semantic (CFG/DDG level) features, then merges them through a "Gated Merging" layer.
* **Evaluation :** The model can be tested under two scenarios described in the study :
  * **WPDP** (Within-Project Defect Prediction) : Prediction within the same project.
  * **CPDP** (Cross-Project Defect Prediction) : Prediction across projects.

The final results (Precision, Recall, F1-Score and AUC) will determine whether the inspected file is classified as "Defective" or "Healthy".