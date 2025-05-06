import os
from pathlib import Path
import re
import numpy as np

path_to_groundtruth = "/home/jshin/csc200/lotte-500-gnd-results"
path_to_test_file = "/home/jshin/csc200/lotte-500-gnd-parallel-results_sorted/lotte-500-gnd-equiwidth-sorted-s-results/Result"
groundtruth_filename = "lotte-500-gnd-groundtruth-top5--.tsv"
test_file_name = "MUVERA_top-5_equiwidth_sorted_result.tsv"

q = 100
k = 5

ground_truth_file = re.sub('\\t', ',', Path(path_to_groundtruth, groundtruth_filename).read_text()).split("\n")
test_file =  re.sub('\\t', ',', Path(path_to_test_file, test_file_name).read_text()).split("\n")

queries = []
q_check = []
current = []
check = []
count = 0
for index in range(len(ground_truth_file)):
    if ground_truth_file[index] != "":
        if count == k:
            q_check.append(check)
            check = []
            count = 0
        check.append(int(ground_truth_file[index].split(",")[1]))
        count = count + 1
count = 0
qcount = 0
for index in range(len(test_file)):
    if test_file[index] != "":
        if count == k:
            queries.append(current)
            current = []
            count = 0
            qcount = qcount + 1
        if qcount < len(q_check) and int(test_file[index].split(",")[1]) in q_check[qcount]:
            current.append(1)
        else:
            current.append(0)
        count = count + 1

prescision_q_results = []
recall_q_results = []
f1_q_results = []
f05_q_results = []
for n in queries:
    prescision_results = []
    recall_results = []
    f1_results = []
    f05_results = []
    count = 1
    while count <= k:
        if sum(n) == 0:
            prescision_results.append(0)
            recall_results.append(0)
            f1_results.append(0)
            f05_results.append(0)
        else:
            p = sum(n[0:count])/count
            r = sum(n[0:count])/sum(n)
            prescision_results.append(p)
            recall_results.append(r)
            if p+r == 0:
                f1_results.append(0)
            else:
                res = ((2)*p*r)/(p+r)
                f1_results.append(res)
            if (0.5*0.5)*p + r == 0:
                f05_results.append(0)
            else:
                res = ((1+(0.5*0.5))*p*r)/((0.5*0.5)*p + r)
                f05_results.append(res)
        count = count + 1
    prescision_q_results.append(prescision_results)
    recall_q_results.append(recall_results)
    f1_q_results.append(f1_results)
    f05_q_results.append(f05_results)
p_sum = np.sum(prescision_q_results, axis = 0)/len(queries)
r_sum = np.sum(recall_q_results, axis = 0)/len(queries)
f1_sum = np.sum(f1_q_results, axis = 0)/len(queries)
f05_sum = np.sum(f05_q_results, axis = 0)/len(queries)

open("stats_for_top" + str(k) + ".txt", 'w').close()
f = open("stats_for_top" + str(k) + ".txt", "a")
f.write("prescision@ average results:\n") 
for i in p_sum:
    f.write(str(i) + " ")
f.write("\n\n")
f.write("recall@ average results:\n") 
for i in r_sum:
    f.write(str(i) + " ")
f.write("\n\n")
f.write("F1@ average results:\n") 
for i in f1_sum:
    f.write(str(i) + " ")
f.write("\n\n")
f.write("F0.5@ average results:\n") 
for i in f05_sum:
    f.write(str(i) + " ")
f.write("\n\n")

f.write("prescision@ results:\n") 
for i in prescision_q_results:
    for n in i:
        f.write(str(n) + " ")
    f.write("\n")
f.write("\n")
f.write("recall@ results:\n")
for i in recall_q_results:
    for n in i:
        f.write(str(n) + " ")
    f.write("\n")
f.write("\n")
f.write("f1@ results:\n")
for i in f1_q_results:
    for n in i:
        f.write(str(n) + " ")
    f.write("\n")
f.write("\n")
f.write("f0.5@ results:\n")
for i in f05_q_results:
    for n in i:
        f.write(str(n) + " ")
    f.write("\n")
f.write("\n")
f.close()





