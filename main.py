import os
import pickle

import shap.maskers
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np

data_dir = './dat'

data = {}

# with open('training_data_2048.bin', 'rb') as f:
#     data['training_data'] = np.fromfile(f, dtype=np.dtype((np.float32, (2048,))))

with open('validation_data_2048.bin', 'rb') as f:
    data['validation_data'] = np.fromfile(f, dtype=np.dtype((np.float32, (2048,))))

# with open('training_labels.bin', 'rb') as f:
#     data['training_labels'] = np.fromfile(f, dtype=np.uint16)

with open('validation_labels.bin', 'rb') as f:
    data['validation_labels'] = np.fromfile(f, dtype=np.uint16)

# pipeline = None
# print()
# # train model
# pipeline = make_pipeline(StandardScaler(), MLPClassifier(solver='adam', hidden_layer_sizes=(1024, 218)))
# pipeline.fit(data['training_data'], data['training_labels'])

# with open('./model_mlp_2048.p', 'wb') as f:
#     pickle.dump(pipeline, f)

with open('./model_mlp_0.39_2048.p', 'rb') as f:
    pipeline = pickle.load(f)

# evaluate the classifier

print("[INFO] evaluating classifier...")
predictions = pipeline.predict(data['validation_data'])

# score = accuracy_score(predictions, data['validation_labels'])
# print(score)

print(classification_report(data['validation_labels'], predictions))

# model = RandomForestClassifier(random_state=0)
# model.fit(data['training_data'], data['training_labels'])

# # test performance
# y_pred = model.predict(data['validation_data'])

# save the model
