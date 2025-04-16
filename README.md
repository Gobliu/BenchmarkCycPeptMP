# BenchmarkCycPeptMP
PyTorch implementation for *__An extensive benchmark study on membrane permeability prediction of cyclic peptides__* <br />

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

We use the **CycPeptMPDB** dataset consisting of over 7,000 curated cyclic peptides with experimentally measured membrane permeability under different assay conditions. For more details, see the [paper](#)(to be added) and `CSV/` folder.

---

