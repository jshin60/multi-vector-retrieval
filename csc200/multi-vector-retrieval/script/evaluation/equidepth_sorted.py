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
top_k = 10
k_sim = 5
n_candidate = [20]
parallel = False
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
maxlength = int(max(lengths))
documents = []
used_buckets = []
for n in range(maxlength + 1):
    documents.append([])
total_count = 0
for index in range(len(lengths)):
    current_vector_size = lengths[index]
    count = 0
    current_vector = []
    while count < current_vector_size:
        current_vector.append(encodings[total_count + count])
        count = count + 1
    total_count = total_count + count
    documents[int(lengths[index]%(maxlength + 1))].append(document(current_vector, index, lengths[index], rawdata[index]))
    if int(lengths[index]%(maxlength + 1)) not in used_buckets:
        used_buckets.append(int(lengths[index]%(maxlength + 1)))
used_buckets = sorted(used_buckets)
end_time = time.perf_counter()
elapsed_extract_time = end_time - start_time

#Partition data
start_time = time.perf_counter()
splits = int(len(used_buckets) / len(partition_names))
remainder = int(len(used_buckets)%len(partition_names))
partition_num = (np.zeros(len(partition_names))) + splits
partition_num[len(partition_names) - 1] = partition_num[len(partition_names) - 1] + remainder
holder = 0
partitions = []
for p in partition_num:
    temp = holder + p
    partition = []
    while holder != temp:
        partition = partition + documents[used_buckets[holder]]
        holder = holder + 1
    partitions.append(partition)
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
        build_index_path = f'{path_to_code}multi-vector-retrieval/script/data'
        #os.system(f'cd {build_index_path} && python3.10 build_index.py {dataset_name}-{partition_names[n]}')
        Muvera(dataset_name+"-"+partition_names[n] + " " + str(top_k) + " " + str(partition_d_proj[n]) + " " + str(partition_r_reps[n]))
if parallel == True:
    for m in range(len(partition_names)):
        n.join()

#Concatenating Datapoints by re-evaluation
start_time = time.perf_counter()
sum_of_re_evaluation = 0
re_evaluation_dataset = []
for n in n_candidate:
    for partition in range(len(partition_names)):
        rankings = f'{dataset_name}-{partition_names[partition]}-MUVERA-top{top_k}-k_sim_{k_sim}-d_proj_{partition_d_proj[partition]}-r_reps_{partition_r_reps[partition]}-n_candidate_{n}.tsv'
        ranking_file = re.sub('\\t', ',', Path(f'{path_to}/multi-vector-retrieval/Result/answer/', rankings).read_text()).split('\n')
        for ranking in ranking_file:
            if ranking != "" and partitions[partition][int(ranking.split(",")[1])].taken != True:
                re_evaluation_dataset.append(partitions[partition][int(ranking.split(",")[1])])
                partitions[partition][int(ranking.split(",")[1])].taken = True
                sum_of_re_evaluation = sum_of_re_evaluation + 1
toencode = []
todoclen = []
collection = ""
for index in re_evaluation_dataset:
    toencode = toencode + index.vector
    todoclen.append(index.size)
    collection = collection + index.text + "\n"
np.save(os.path.join(f'{path_to}/multi-vector-retrieval/Embedding/{dataset_name}-re', 'base_embedding', f'encoding0_float32.npy'), toencode)
np.save(os.path.join(f'{path_to}/multi-vector-retrieval/Embedding/{dataset_name}-re', 'base_embedding', f'doclens0.npy'), todoclen)
np.save(os.path.join(f'/{path_to}/multi-vector-retrieval/Embedding/{dataset_name}-re', f'doclens.npy'), todoclen)
#Can comment out these three lines if one does not want to store the raw data of the points twice and alreayd has ground truths
f = open(os.path.join(f'{path_to}/multi-vector-retrieval/RawData/{dataset_name}-re/document', f'collection.tsv'), "w")
f.write(collection)
f.close()
end_time = time.perf_counter()
elapsed_concatenation_time = end_time - start_time

#Re-evaluation + Formatting
build_index_path = f'{path_to_code}multi-vector-retrieval/script/data'
os.system(f'cd {build_index_path} && python3.10 build_index.py {dataset_name}-re')
Muvera(dataset_name+"-re" + " " + str(top_k) + " " + str(partition_d_proj[len(partition_d_proj)-1]) + " " + str(partition_r_reps[len(partition_r_reps)-1]))
start_time = time.perf_counter()
for n in n_candidate:
    rankings = f'{dataset_name}-re-MUVERA-top{top_k}-k_sim_{k_sim}-d_proj_{partition_d_proj[len(partition_d_proj)-1]}-r_reps_{partition_r_reps[len(partition_r_reps)-1]}-n_candidate_{n}.tsv'
    ranking_file = re.sub('\\t', ',', Path(f'{path_to}/multi-vector-retrieval/Result/answer/', rankings).read_text()).split('\n')
    new_ranking_file = ""
    for ranking in ranking_file:
        if ranking != '':
            temp = ranking.split(",")
            new_val = str(re_evaluation_dataset[int(temp[1])].index)
            temp[1] = new_val
            new_item = ""
            for t in range(len(temp)):
                if t != len(temp) - 1:
                    new_item = new_item + temp[t] + "\t"
                else:
                    new_item = new_item + temp[t]
            new_ranking_file = new_ranking_file + new_item + "\n"
    f = open(os.path.join(f'{path_to}/multi-vector-retrieval/Result/answer/', rankings), "w")
    f.write(new_ranking_file)
    f.close()
end_time = time.perf_counter()
elapsed_indexing_time = end_time - start_time
open(str(dataset_name) + "_top" + str(top_k) + "_equidepth_sorted_time.txt", 'w').close()
f = open(str(dataset_name) + "_top" + str(top_k) + "_equidepth_sorted_time.txt", "a")
f.write("Number of datapoints in combination consideration: " + str(sum_of_re_evaluation) +"\n" +
"Elapsed extraction time: " + str(elapsed_extract_time) + "\n" +
"Elapsed partition time: " + str(elapsed_partition_time) + "\n" +
"Elapsed write time: " + str(elapsed_write_time) + "\n"+
"Elapsed recombination time: " + str(elapsed_concatenation_time) + "\n"+
"Elapsed Final Indexing time: " + str(elapsed_indexing_time) + "\n")
f.close()