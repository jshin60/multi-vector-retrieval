import matplotlib.pyplot as plt
import re
import sys

inputfile = sys.argv[1]

f = open(inputfile, "r")
n_item_dimensions_string = f.read()
temp = n_item_dimensions_string.split(",")
n_item = int(temp[0])
n_item_dimensions_string = temp[1]
n_item_dimensions_string = re.sub(',,', ',', re.sub('\s', ',', re.sub('\\n|\[|\]', '', n_item_dimensions_string))).split(",")
item_dims = []
for n in n_item_dimensions_string:
    if n != "":
        item_dims.append(int(n))
fig, ax = plt.subplots()
ax.hist(item_dims, bins=int(n_item/10), linewidth=0.5)
plt.title("Distribution of Vector Lengths of " + inputfile)
plt.xlabel("Vector Lengths")
plt.ylabel("Frequency")
plt.savefig(re.sub('.txt', '_', inputfile) + "plot" + ".png")