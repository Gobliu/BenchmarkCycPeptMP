# BenchmarkCycPeptMP
PyTorch implementation for *__Systematic benchmarking of 13 AI methods for predicting cyclic peptide membrane permeability__* <br />

Wei Liu, Jianguo Li, Chandra S. Verma and Hwee Kuan Lee*


## 🧬 Overview

Cyclic peptides are promising drug candidates due to their ability to target protein–protein interactions. However, their limited membrane permeability remains a major hurdle for intracellular applications. 

In this benchmark study, we systematically evaluate **13 machine learning and deep learning models** for cyclic peptide permeability prediction. These models span **four types of molecular representations**:  
- Fingerprints  
- Strings (e.g., SMILES)  
- Graphs  
- Images  

We test these models across:
- **Three tasks**: regression, binary classification, and soft-label classification  
- **Two data-splitting strategies**: random split and scaffold split  

Our results show that:
- **Graph-based models**, particularly **Directed Message Passing Neural Network (DMPNN)**, consistently achieve top performance.  
- **Regression tasks** generally outperform classification.  
- **Scaffold split**, although better for generalization assessment, leads to lower predictive accuracy than random split.  
- Current models approach the variability of experimental measurements, indicating strong practical value, while still leaving room for further improvement.

---
## 📦 Requirements

This project has been tested with the following versions:

- `DeepChem==2.7.1`  
- `RDKit==2022.09.4`  
- `PyTorch==2.0.1`

The current version of `DeepChem` does not support classification tasks with soft labels, we modified
function `SparseSoftmaxCrossEntropy` in `/YOURPATH/deepchem/models/losses.py` (line 294) from:
```
return ce_loss(output, labels.long())
```
to

```
if len(labels.shape) == len(output.shape)  and labels.size(-1) == output.size(-1):
    return ce_loss(output, labels)
else:
    return ce_loss(output, labels.long())
```
---

## 🧪 Models Evaluated

### 📁 `ClassicalML/`
- **RF** – Random Forest  
- **SVM** – Support Vector Machine

### 📁 `DeepChemModels/`
- **AttentiveFP**   
- **DMPNN** – Directed Message Passing Neural Network  
- **GAT** – Graph Attention Network  
- **GCN** – Graph Convolutional Network  
- **MPNN** – Message Passing Neural Network  
- **PAGTN** – Path-Augmented Graph Transformer Network  
- **ChemCeption**
### 📁 `PyTorchModels/`
- **RNN** – Recurrent Neural Network  
- **LSTM** – Long Short-Term Memory Network  
- **GRU** – Gated Recurrent Unit

---

## 📂 Dataset

We use the **CycPeptMPDB** dataset consisting of over 7,000 curated cyclic peptides with experimentally measured membrane permeability under different assay conditions. For more details, see the [paper](#https://doi.org/10.1186/s13321-025-01083-4) and `CSV/` folder.

---

## 🚀 How to Train a Model

There are **four ways** to train a model in this repository:

---

### 1. Use `DispatcherMain.py` with `Config.yaml` (**Recommended**)

Edit `Config.yaml` to specify the training configuration, such as:

```yaml
model: RNN
mode: classification
split: random
n_epoch: 1000
```
Them run:
```
python DispatcherMain.py
```
Automatically detects the model type and dispatches to the appropriate main script.


### 2. Run a main script directly with `Config.yaml`

Each main script will read `Config.yaml` as the default:

```bash
python ClassicalML/MLMain.py
```
or
```bash
python DeepChemModels/DeepChemModelsMain.py
```
or
```bash
python PytorchModels/PytorchModelsMain.py
```
Works the same as DispatcherMain but calls the script explicitly.

### 3. Use `DispatcherMain.py` and override `Config.yaml` via command-line

You can override settings defined in `Config.yaml` by passing command-line arguments:

```bash
python DispatcherMain.py --model GRU --mode regression --split scaffold --n_epoch 2000
```

### 4. Run a main script and override `Config.yaml` via command-line

Same idea as (3), but calls the main script explicitly:

```bash
python DeepChemModels/DeepChemModelsMain.py --model=DMPNN --split=random --mode=soft --batch_size=128
```
