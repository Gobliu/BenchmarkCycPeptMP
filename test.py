import deepchem as dc
import numpy as np
import torch
import os
from deepchem.models import DMPNNModel
from deepchem.feat.smiles_tokenizer import SmilesTokenizer, get_default_tokenizer

# Load Tox21 dataset
# tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(splitter='scaffold')
#
# train_dataset, valid_dataset, test_dataset = tox21_datasets
# print('dataset is featurized')
#
# # Assess featurized data
# print(len(train_dataset), len(valid_dataset), len(test_dataset))
# print(train_dataset.X[:5])
#
# # Initialise the model
# model = DMPNNModel(n_tasks=len(tox21_tasks), \
#                    n_classes=2, \
#                    mode='classification', \
#                    batch_size=50, \
#                    global_features_size=200)
#
# # Model training
# print("Training model")
# model.fit(train_dataset, nb_epoch=30)
#
# # Model evaluation
# metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
#
# print("Evaluating model")
# train_scores = model.evaluate(train_dataset, [metric], transformers)
# valid_scores = model.evaluate(valid_dataset, [metric], transformers)
#
# print("Train scores: ", train_scores)
# print("Validation scores: ", valid_scores)

# loss_function = dc.models.losses.SparseSoftmaxCrossEntropy()
# pytorch_loss = loss_function._create_pytorch_loss()
#
# true = torch.ones((2, 3)).float()
# pred = torch.ones((2, 3)).float()
# print(pytorch_loss(true, pred))

# inputs = torch.tensor([[2.3, 2.1]])
#
# target = torch.tensor([[0.9, 0.1]]).float()
# # one_hot_target = torch.tensor([[0,1,0], [0,0,1]]).float()
#
# # weights = torch.tensor([25., 25., 100.])
#
# ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
# print(ce_loss(inputs, target))
#
# softmax = torch.exp(inputs) / torch.sum(torch.exp(inputs))
# print(softmax)
# print(-torch.sum(torch.log(softmax) * target))
# nll_loss = torch.nn.NLLLoss(reduction='mean')
# print(nll_loss(torch.log(softmax), target))
#
# current_dir = os.path.dirname(os.path.realpath(__file__))
# # vocab_path = os.path.join('/')
#
# # from deepchem.feat.molecule_featurizers.smiles_to_seq import SmilesToSeq
# # print(vocab_path)
tokenizer = SmilesTokenizer('./vocab.txt')
print(tokenizer.encode("CC(=O)OC1=CC=CC=C1C(=O)O"))

# from transformers import BertTokenizerFast, BertModel
#
# checkpoint = "unikeibert-base-smiles"  # The pretrained model checkpoint
# tokenizer = BertTokenizerFast.from_pretrained(checkpoint)  # Load the tokenizer
# model = BertModel.from_pretrained(checkpoint)  # Load the model
#
# # Example SMILES string
# example = "O=C([C@@H](c1ccc(cc1)O)N)N[C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C"
# tokens = tokenizer(example, return_tensors="pt")  # Tokenize the SMILES string
# predictions = model(**tokens)  # Generate predictions


import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(0)

# Define the input dimensions
input_size = 5    # Number of features in the input
hidden_size = 10  # Number of features in the hidden state
num_layers = 1    # Number of recurrent layers

# Create an RNN model
rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

# Generate a sample input (batch size of 3, sequence length of 4, and input size of 5)
# The shape is (batch_size, sequence_length, input_size)
sample_input = torch.randn(3, 4, input_size)

# Initialize the hidden state (num_layers, batch_size, hidden_size)
hidden_state = torch.zeros(num_layers, 3, hidden_size)

# Run the RNN model
output, hn = rnn(sample_input, hidden_state)

# Print the output and the final hidden state
print("Output shape:", output.shape)
print("Output:", output)
print("Hidden state shape:", hn.shape)
print("Hidden state:", hn)
