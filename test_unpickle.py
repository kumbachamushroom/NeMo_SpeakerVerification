import torch
import pickle
import os

with open(os.path.join(os.getcwd(), 'embeddings', 'track_manifest_embeddings.pkl'), 'rb') as f:
    data = pickle.load(f).items()
    embs = [emb for _,emb in data][:10]
print(len(embs))
#data = pickle.load('target_embeddings.pkl')
#for i, v in data.iteritems():
#    print(i,v)