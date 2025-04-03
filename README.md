# BenchmarkCycPeptMP
PyTorch implementation for *__An extensive benchmark study on membrane permeability prediction of cyclic peptides__* <br />

Wei Liu, Jianguo Li, Chandra S. Verma and Hwee Kuan Lee*

# Abstract
Motivation: Cyclic peptides are promising drug candidates due to their ability to target protein-protein interactions; however, their limited membrane permeability hinders their effectiveness in intracellular applications. Accurate
permeability prediction can facilitate the identification and optimization of cell-permeable cyclic peptides, accelerating drug development. Deep learning models have shown promise in predicting molecular properties, yet their effectiveness
in permeability prediction remains underexplored. A systematic evaluation of these models is essential to assess current capabilities and guide future developments.

Results: We present a comprehensive benchmark study evaluating 13 machine learning and deep learning models for cyclic peptide permeability prediction, spanning four molecular representations: fingerprints, strings, graphs, and images.
Using the CycPeptMPDB dataset, we assess model performance across three tasks—regression, binary classification, and soft-label classification—under two data-splitting strategies: random split and scaffold split. Our results show that graph-
based models, particularly the Directed Message Passing Neural Network (DMPNN), achieve the highest performance in most settings. Regression tasks generally outperform classification approaches, and scaffold split validation, although
designed to assess generalization, leads to significantly lower predictive performance compared to random split validation. Comparison with experimental variability highlights the practical value of current models, while also indicating room for
further improvement.

# Requirements:
* torch 2.1.0
* torch_geometric 2.4.0
* RDKit 2023.03.3
