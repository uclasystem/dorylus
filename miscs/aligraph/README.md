# Running AliGraph Experiments

## Configuring the cluster
Make sure that each of the nodes is configured the same such that the
directory structure of each is identical.
Any of the nodes can be the 'main' node.
Put a file in the home directory called `group_ips` that will be used to run
the experiments and communicate updates.

Make sure to pip install `graph-learn` as mentioned
[here](https://github.com/alibaba/graph-learn/blob/master/docs/install_en.md).

Make sure that there is some networked file system accessible by all
nodes, such as NFS or HDFS.
For each dataset, prepare it according to the instructions at the following
[link](https://github.com/alibaba/graph-learn/blob/master/docs/quick_start_en.md#31-data-preparation).
If the data is already in Dorylus's format, the `convert-to-aligraph.py`
script will do this automatically with the following:
```
python convert-to-aligraph.py [dataset folder]
```
where `dataset foler` contains the files `graph.bsnap`, `features.bsnap`,
and `labels.bsnap`.

Finally, specify the cluster configuration in the `run-dist.sh` file.
For example, with a cluster with IPs `172.31.1.1, 172.31.2.2, 172.31.3.3`,
the config would look like
```
--ps_hosts=172.31.1.1:2222 \
--worker_hosts=172.31.1.1:2223,172.31.2.2:2223,172.31.3.3:2223 \
```

AliGraph uses a `tracker` dir to find endpoints of its workers.
Make sure to modify the `TRACKER` variable in both `run-all.sh` and `run-dist.sh`.

Once this is done, synchronize the configuration with the other nodes by
running `for-all.sh send -f run-dist.sh`.
Note that we only need to send `run-dist.sh` because we will only execute
`run-all.sh` from the main node.

The `for-all.sh` script can be used to send files to all nodes or run commands
on all other nodes with the following syntax:
```
for-all.sh do -c "some command in quotes"
for-all.sh do -f "a list of files in quotes"
```

## Running AliGraph
Now that the servers are configured, all you have to do is run the `run-all.sh`
script with the following usage:
```
run-all.sh <dataset> [# workers per node]
```

Here, `dataset` is the folder name containing the dataset in the networked file
system, such as `/mnt/[dataset]`.
In addition, there should exist some config in the configs directory called
`configs/[dataset]_config.json`.
For examples on the options you can provide to a config, see the examples.

If you update a configuration or cluster configuraiton, make sure to sync it
across all nodes using the `for-all.sh` script.

## Profiling Execution
Just a note that in order to profile the epoch times you either need to sum
the output times from the log or replace the DistTFTrainer from the graph-learn
repo with a version that tracks epoch times.
To do this just grab the file `graph-learn/graphlearn/python/model/tf/trainer.py`
from the AliGraph repo and use your local version.
