# -*- coding: utf-8 -*-

import os, json
import numpy as np
from extract_feature import BertVector
from keras.models import load_model
from att import Attention
from utils.crf import CRF

# Load the best trained model
# model_dir = '/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/output'
# files = os.listdir(model_dir)
# models_path = [os.path.join(model_dir, _) for _ in files]
# best_model_path = sorted(models_path, key=lambda x: float(x.split('-')[-1].replace('.hdf5', '')), reverse=True)[0]
# print(best_model_path)
# model = load_model(best_model_path, custom_objects={"Attention": Attention})
model = load_model(
  "output/oversampling/e40-dr03-b64-dim256/e-38-los-0.4263-acc-0.8790.hdf5", 
  custom_objects={"CRF": CRF})

# Example Statements and Preprocessing
#text1 = 'green tea#skin papillomas#Inhibitory effect of green tea on the growth of established skin papillomas in mice.'
text1 = 'tobacco#lung cancer#In conclusion, variation in lung cancer rates between populations may be due to: (1) differences in effective exposure to tobacco smoke carcinogens; (2) differences in factors which modify the effect of tobacco smoke, including differences in host susceptibility and metabolism of carcinogens, or (3) differences in exposure to other independent risk factors for lung cancer.'
per1, per2, doc = text1.split('#')
text = '$'.join([per1, per2, doc.replace(per1, len(per1)*'#').replace(per2, len(per2)*'#')])
print(text)


bert_model = BertVector(pooling_strategy="NONE", max_seq_len=512)
vec = bert_model.encode([text])["encodes"][0]
x_train = np.array([vec])

predicted = model.predict(x_train)
y = np.argmax(predicted[0])

with open('input/rel_dict.json', 'r', encoding='utf-8') as f:
    rel_dict = json.load(f)

id_rel_dict = {v:k for k,v in rel_dict.items()}
print('original: %s' % text1)
print('Predict entity relationships: %s' % id_rel_dict[y])
