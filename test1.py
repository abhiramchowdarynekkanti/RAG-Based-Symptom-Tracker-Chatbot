from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
import torch
import faiss
import numpy as np
import pandas as pd
import pickle
import re

tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

dataset = pd.read_csv("Dataset/Symptom2Disease.csv", usecols=['label','text'])
dataset = dataset.values
print(dataset.shape)

X = []
Y = []
for i in range(len(dataset)):
    try:
        disease = dataset[i,0]
        data = dataset[i,1].strip('\n').strip().lower()
        data = re.sub('[^a-z]+', ' ', data)
        inputs = tokenizer(data, return_tensors="pt")
        input_ids = inputs["input_ids"]
        symptoms_hidden_states = model.question_encoder(input_ids)[0]
        symptoms_hidden_states = symptoms_hidden_states.detach().numpy().ravel()
        X.append(symptoms_hidden_states)
        Y.append(disease)
        print(str(i)+" "+str(len(data)))
    except:
        print(data+"===============================================================")
        pass
X = np.asarray(X)
print(X.shape)
dimension = X.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(X)
print(faiss_index)

f = open('model/faiss.pckl', 'wb')
pickle.dump(faiss_index, f)
f.close()
Y = np.asarray(Y)
np.save("model/Y", Y)

f = open('model/faiss.pckl', 'rb')
index = pickle.load(f)
f.close()
Y = np.load("model/Y.npy")

query = "skin around my mouth, nose, and eyes is red and inflamed"
inputs = tokenizer(query, return_tensors="pt")
input_ids = inputs["input_ids"]
query = model.question_encoder(input_ids)[0]
query = query.detach().numpy()

distances, indices = faiss_index.search(query, k=3) 
# Retrieve original texts based on indices

for i, idx in enumerate(indices[0]):
    print(str(distances[0][i])+" "+Y[idx])





                
