import random


dataset_name = "large"
num_vertices = 120000
num_features = 602
label_kinds = 41
edge_probability = 0.0005
dataset_dir = "../../../data/" + dataset_name + "/"


# Generate graph.
with open(dataset_dir + dataset_name + ".graph", 'w+') as fgraph:
    for i in range(num_vertices):
        for j in range(num_vertices):
            if random.random() < edge_probability:
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
