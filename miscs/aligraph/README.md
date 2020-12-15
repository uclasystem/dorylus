# Running AliGraph Experiments

## Configuring the cluster
Make sure that each of the nodes is configured the same such that the
directory structure of each is identical.
Any of the nodes can be the 'main' node.
Put a file in the home directory called `group_ips` that will be used to run
the experiments and communicate updates.

Make sure to pip install `graph-learn` as mentioned
[here](https://github.com/alibaba/graph-learn/blob/master/docs/install_en.md).

Finally make sure that there is some networked file system accessible by all
nodes, such as NFS or HDFS.
For each dataset, prepare it according to the instructions at the following
[link](https://github.com/alibaba/graph-learn/blob/master/docs/install_en.md).


## Running AliGraph
Now that the server is configured, all you have to do is run the `run-all.sh`
script with the following usage:
```
run-all.sh <dataset> [# workers per node]
```
