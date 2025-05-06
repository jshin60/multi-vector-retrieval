import multiprocessing
import os
from pathlib import Path
import numpy as np
import re
import time

#inputs
dataset_name = 'lotte-500-gnd'
#CHANGE TO YOUR PATH
path_to = '/home/jshin/csc200/Dataset'
path_to_code = '/home/jshin/csc200/'
partition_names = ['p1', 'p2', 'p3', 'p4', 'p5']
partition_d_proj = [8, 8, 8, 8, 8, 8]
partition_r_reps = [20, 20, 20, 20, 20, 20]
top_k = 5
k_sim = 5
q = 100
n_candidate = [20]
parallel = True
build_indexes = False

#Document information class
class document():
    def __init__(self,vector,index, size, text):
        self.vector = vector
        self.index = index
        self.size = size
        self.text = text
        self.taken = False
  
#Extract data 
start_time = time.perf_counter()
embedding_dir = f'{path_to}/multi-vector-retrieval/Embedding/{dataset_name}'
encodings = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy'))
lengths = np.load(os.path.join(embedding_dir, f'doclens.npy'))
rawdata = Path(f'{path_to}/multi-vector-retrieval/RawData/{dataset_name}/document', f'collection.tsv').read_text().split("\n")
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
end_time = time.perf_counter()
elapsed_extract_time = end_time - start_time

#Partition data
start_time = time.perf_counter()
splits = int(len(documents) / len(partition_names))
remainder = int(len(documents)%len(partition_names))
partition_num = (np.zeros(len(partition_names))) + splits
partition_num[len(partition_names) - 1] = partition_num[len(partition_names) - 1] + remainder
holder = 0
partitions = []
for p in partition_num:
    temp = int(holder)
    holder = p + holder
    partitions.append(documents[temp:int(holder)])
end_time = time.perf_counter()
elapsed_partition_time = end_time - start_time

#Write data
start_time = time.perf_counter()
for partition in range(len(partition_names)):
    toencode = []
    todoclen = []
    collection = ""
    for index in partitions[partition]:
        toencode = toencode + index.vector
        todoclen.append(index.size)
        collection = collection + index.text + "\n"
    np.save(os.path.join(f'{path_to}/multi-vector-retrieval/Embedding/{dataset_name}-{partition_names[partition]}', 'base_embedding', f'encoding0_float32.npy'), toencode)
    np.save(os.path.join(f'{path_to}/multi-vector-retrieval/Embedding/{dataset_name}-{partition_names[partition]}', 'base_embedding', f'doclens0.npy'), todoclen)
    np.save(os.path.join(f'/{path_to}/multi-vector-retrieval/Embedding/{dataset_name}-{partition_names[partition]}', f'doclens.npy'), todoclen)
    #Can comment out these three lines if one does not want to store the raw data of the points twice and alreayd has ground truths
    f = open(os.path.join(f'{path_to}/multi-vector-retrieval/RawData/{dataset_name}-{partition_names[partition]}/document', f'collection.tsv'), "w")
    f.write(collection)
    f.close()
end_time = time.perf_counter()
elapsed_write_time = end_time - start_time

#Muvera Processing
def Muvera(arguments):
    os.system('python3.10 eval_muvera.py ' + arguments)

if build_indexes:
    for n in range(len(partition_names)):
        build_index_path = f'{path_to_code}multi-vector-retrieval/script/data'
        os.system(f'cd {build_index_path} && python3.10 build_index.py {dataset_name}-{partition_names[n]}')

for n in range(len(partition_names)):
    if parallel == True:
        n = multiprocessing.Process(target=Muvera, args = {dataset_name+"-"+partition_names[n] + " " + str(top_k) + " " + str(partition_d_proj[n]) + " " + str(partition_r_reps[n])})
        n.start()
    else:
        Muvera(dataset_name+"-"+partition_names[n] + " " + str(top_k) + " " + str(partition_d_proj[n]) + " " + str(partition_r_reps[n]))
if parallel == True:
    for m in range(len(partition_names)):
        n.join()

#Concatenating Datapoints
start_time = time.perf_counter()
sum_of_re_evaluation = 0
result = []
queries = []
first = True
for n in range(q):
    result.append([])
    queries.append(0)
for n in n_candidate:
    for partition in range(len(partition_names)):
        rankings = f'{dataset_name}-{partition_names[partition]}-MUVERA-top{top_k}-k_sim_{k_sim}-d_proj_{partition_d_proj[partition]}-r_reps_{partition_r_reps[partition]}-n_candidate_{n}.tsv'
        ranking_file = re.sub('\\t', ',', Path(f'{path_to}/multi-vector-retrieval/Result/answer/', rankings).read_text()).split('\n')
        count = 0
        query = 0
        queries[0] = ranking_file[0].split(",")[0]
        for ranking in ranking_file:
            if ranking != "":
                if count == top_k:
                    query = query + 1
                    count = 0
                    queries[query] = ranking.split(",")[0]
                result[query].append([partitions[partition][int(ranking.split(",")[1])], float(ranking.split(",")[3])])
                count = count + 1
result_string = ""
for n in range(q):
    temp = sorted(result[n], key=lambda x: x[1], reverse=True)
    temp = temp[:top_k]
    count = 1
    for items in temp:
        current = queries[n]
        current_index = items[0].index
        result_string = result_string + current + "\t" + str(current_index) + "\t" + str(count) + "\t" + str(items[1]) + "\n"
        count = count + 1
f = open("MUVERA_top-" + str(top_k) + "_equiwidth_result.tsv", "w")
f.write(result_string)
f.close()
end_time = time.perf_counter()
elapsed_concatenation_time = end_time - start_time

open(str(dataset_name) + "_top" + str(top_k) + "_equiwidth_time.txt", 'w').close()
f = open(str(dataset_name) + "_top" + str(top_k) + "_equiwidth_time.txt", "a")
f.write("Number of datapoints in combination consideration: " + str(sum_of_re_evaluation) +"\n" +
"Elapsed extraction time: " + str(elapsed_extract_time) + "\n" +
"Elapsed partition time: " + str(elapsed_partition_time) + "\n" +
"Elapsed write time: " + str(elapsed_write_time) + "\n"+
"Elapsed recombination time: " + str(elapsed_concatenation_time) + "\n")
f.close()