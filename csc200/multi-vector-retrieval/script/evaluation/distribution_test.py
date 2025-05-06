import os
from pathlib import Path
import numpy as np

class document():
    def __init__(self,vector,index, size, text):
        self.vector = vector
        self.index = index
        self.size = size
        self.text = text

embedding_dir = f'/home/jshin/csc200/Dataset/multi-vector-retrieval/Embedding/lotte-500-gnd'
encodings = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy'))
lengths = np.load(os.path.join(embedding_dir, f'doclens.npy'))
rawdata = Path(f'/home/jshin/csc200/Dataset/multi-vector-retrieval/RawData/lotte-500-gnd/document', f'collection.tsv').read_text().split("\n")
documents = []
total_count = 0
for index in range(len(lengths)):
    current_vector_size = lengths[index]
    count = 0
    current_vector = []
    while count < current_vector_size:
        current_vector.append(encodings[total_count + count])
        count = count + 1
    total_count = total_count + count
    documents.append(document(current_vector, index, lengths[index], rawdata[index]))
documents = sorted(documents, key=lambda document: document.size)
toencode = []
todoclen = []
collection = ""
for index in documents:
    toencode = toencode + index.vector
    todoclen.append(index.size)
    collection = collection + index.text + "\n"
np.save(os.path.join(f'/home/jshin/csc200/Dataset/multi-vector-retrieval/Embedding/lotte-500-gnd-reduced', 'base_embedding', f'encoding0_float32.npy'), toencode)
np.save(os.path.join(f'/home/jshin/csc200/Dataset/multi-vector-retrieval/Embedding/lotte-500-gnd-reduced', 'base_embedding', f'doclens0.npy'), todoclen)
np.save(os.path.join(f'/home/jshin/csc200/Dataset/multi-vector-retrieval/Embedding/lotte-500-gnd-reduced', f'doclens.npy'), todoclen)
f = open(os.path.join(f'/home/jshin/csc200/Dataset/multi-vector-retrieval/RawData/lotte-500-gnd-reduced/document', f'collection.tsv'), "w")
f.write(collection)
f.close()