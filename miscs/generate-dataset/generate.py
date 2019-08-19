import random


dataset_dir = "../../../data/large/"
num_vertices = 30000
num_features = 60
label_kinds = 12


# Generate graph.
with open(dataset_dir + "large.graph", 'w+') as fgraph:
    for i in range(num_vertices):
        for j in range(num_vertices):
            if random.random() > 0.99:
                fgraph.write(str(i) + " " + str(j) + "\n")

print("Graph done.")

# Generate initial features.
with open(dataset_dir + "features", 'w+') as ffeats:
    for i in range(num_vertices):
        for j in range(num_features - 1):
            ffeats.write(str(random.random()) + ", ")
        ffeats.write(str(random.random()) + "\n")

print("Features done.")

# Generate labels.
with open(dataset_dir + "labels", 'w+') as flabels:
    for i in range(num_vertices):
        flabels.write(str(random.randint(0, label_kinds - 1)) + "\n")

print("Labels done.")
