import os
import numpy as np
import multiprocessing

class datapoint():
    def __init__(self, vector, index):
        self.vector = vector
        self.index = index

def Test(dataset):
    os.system('python3.10 eval_muvera.py ' + dataset)
for n in range(1):
    n = multiprocessing.Process(target=Test, args = {'lotte-500-gnd-reduced'})
    n.start()
for m in range(1):
    n.join()


