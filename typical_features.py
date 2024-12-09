import os
import pickle

from img2vec_pytorch import Img2Vec

import pandas as pd

from diag_utils import map_diagnos_list

# prepare data

img2vec = Img2Vec(layer_output_size=2048)

data_dir = './dat'
train_dir = os.path.join(data_dir, 'train')

metadata_fields = ['isic_id', 'benign_malignant', 'diagnosis']
train_metadata = pd.read_csv(os.path.join(data_dir, 'train_metadata.csv'), skipinitialspace=True, usecols=metadata_fields )

train_diagnosis = map_diagnos_list(list(zip(train_metadata.benign_malignant, train_metadata.diagnosis)))
train_metadata = list(zip(train_metadata.isic_id, train_diagnosis))

train_diagnosis = set(train_diagnosis)

res = {}
for diag in train_diagnosis:
    for data in train_metadata:
        if data[1] == diag:
            if diag not in res:
                res[diag] = [data[0]]
            else:
                res[diag].append(data[0])
            
            if len(res[diag]) >= 10:
                break

for el in res:
    print(f'DIAGNOSIS {el}\n -> {res[el]}')

with open('model_answers.bin', 'wb') as f:
    pickle.dump(res, f)
