
import deepchem as dc
from deepchem.models import DMPNNModel

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(splitter='scaffold')

train_dataset, valid_dataset, test_dataset = tox21_datasets
print('dataset is featurized')

# Assess featurized data
print(len(train_dataset), len(valid_dataset), len(test_dataset))
print(train_dataset.X[:5])

# Initialise the model
model = DMPNNModel(n_tasks=len(tox21_tasks), \
                   n_classes=2, \
                   mode='classification', \
                   batch_size=50, \
                   global_features_size=200)

# Model training
print("Training model")
model.fit(train_dataset, nb_epoch=30)

# Model evaluation
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores: ", train_scores)
print("Validation scores: ", valid_scores)