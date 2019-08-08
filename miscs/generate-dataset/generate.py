import random


dataset_dir = "../../../data/small/"
num_vertices = 1000
num_features = 6


# Generate graph.
with open(dataset_dir + "small.graph", 'w+') as fgraph:
    for i in range(num_vertices):
        for j in range(num_vertices):
            if random.random() > 0.5:
                fgraph.write(str(i) + " " + str(j) + "\n")


# Generate initial features.
with open(dataset_dir + "features", "w+") as ffeats:
    for i in range(num_vertices):
        for j in range(num_features - 1):
            ffeats.write(str(random.random()) + ", ")
        ffeats.write(str(random.random()) + "\n")
