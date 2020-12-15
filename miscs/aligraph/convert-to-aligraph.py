import os
import struct
import sys
import time
import math
from pathlib import Path

import numpy as np

graph_file_name = 'graph.bsnap'
features_file_name = 'features.bsnap'
labels_file_name = 'labels.bsnap'

TRAIN_PORTION = 0.66
VAL_PORTION = 0.1
TEST_PORTION = 0.24

def timestamp_ms():
    return int(time.time() * 1000)

def parse_header(header):
    vert_type_size = int.from_bytes(header[:4], byteorder='little')
    num_verts = int.from_bytes(header[4:8], byteorder='little')
    num_edges = int.from_bytes(header[8:16], byteorder='little')
    return vert_type_size, num_verts, num_edges

def parse_edge(edge_bin):
    return int.from_bytes(edge_bin[:4], byteorder='little'),\
        int.from_bytes(edge_bin[4:], byteorder='little')

def convert_graph(graph_file):
    ## Output files
    edge_table = "{}/edge_table".format(dataset)
    edge_table_with_self_loop = '{}/edge_table_with_self_loop'.format(dataset)

    print("Processing " + graph_file)
    filesize = Path(graph_file).stat().st_size
    assert(filesize % 8 == 0)
    convert_start = start_read = timestamp_ms()
    edge_no = 0
    idx_set = set()
    with open(graph_file, 'rb') as gf,\
         open(edge_table, 'w') as et,\
         open(edge_table_with_self_loop, 'w') as etwl:
        et.write('\t'.join(["src_id: int64", 'dst_id : int64', 'weight: double']) + '\n')
        etwl.write('\t'.join(["src_id: int64", 'dst_id : int64', 'weight: double']) + '\n')

        gf.read(16)
        nbytes = 16

        while nbytes < filesize:
            nbytes += 8
            src = int.from_bytes(gf.read(4), 'little')
            dst = int.from_bytes(gf.read(4), 'little')
            edge_no += 1

            idx_set.add(src)
            idx_set.add(dst)

            et.write('\t'.join([str(src), str(dst), '0.0']) + '\n')
            etwl.write('\t'.join([str(src), str(dst), '0.0']) + '\n')

            if (timestamp_ms() - start_read >= 10000):
                start_read = timestamp_ms()
                print("Processing edge " + str(edge_no))

        for idx in idx_set:
            etwl.write('\t'.join([str(idx), str(idx), '0.0']) + '\n')

    print("Finished converting graph. Took " + str(timestamp_ms() - convert_start))
    return idx_set

def generate_masks(idx_set):
    train_table = "{}/train_table".format(dataset)
    val_table = "{}/val_table".format(dataset)
    test_table = "{}/test_table".format(dataset)

    idx_list = list(idx_set)
    nverts = len(idx_list)
    train_start = 0
    val_start = math.ceil(nverts * TRAIN_PORTION)
    test_start = math.ceil(val_start + (nverts * VAL_PORTION))

    with open(train_table, 'w') as trainf:
        trainf.write('id:int64\tweight:float\n')
        for i in range(0, val_start):
            trainf.write(str(idx_list[i]) + '\t' + str(1.0) + '\n')

    with open(val_table, 'w') as valf:
        valf.write('id:int64\tweight:float\n')
        for i in range(val_start, test_start):
            valf.write(str(idx_list[i]) + '\t' + str(1.0) + '\n')

    with open(test_table, 'w') as testf:
        testf.write('id:int64\tweight:float\n')
        for i in range(test_start, nverts):
            testf.write(str(idx_list[i]) + '\t' + str(1.0) + '\n')
    

## AliGraph file format combines features and labels (we should have done this...)
def binary_to_feat_vec(num_feats, bin_string):
    feat_vec = [0.0] * num_feats
    for f in range(num_feats):
        feat_vec[f] = struct.unpack('f', bin_string[f * 4: (f+1) * 4])[0]

    feat_vec = [str(f) for f in feat_vec]
    return feat_vec

def convert_node_data(features_file, labels_file):
    ## Output file
    node_table = "{}/node_table".format(dataset)

    feat_file_size = Path(features_file).stat().st_size
    lab_file_size = Path(labels_file).stat().st_size

    with open(features_file, 'rb') as ff,\
         open(labels_file, 'rb') as lf,\
         open(node_table, 'w') as nt:
        nt.write('\t'.join(['id:int64', 'label:int64', 'feature:string']) + '\n')
        num_feats = int.from_bytes(ff.read(4), 'little')
        num_labs = int.from_bytes(lf.read(4), 'little')

        num_verts_feat = (feat_file_size - 4) / num_feats / 4
        num_verts_lab = (lab_file_size - 4) / 4
        assert(num_verts_feat == num_verts_lab)

        nbytes_feat = 4
        nbytes_lab = 4
        idx = 0
        while nbytes_feat < feat_file_size and nbytes_lab < lab_file_size:
            feat_vec = binary_to_feat_vec(num_feats, ff.read(num_feats * 4))
            label = int.from_bytes(lf.read(4), 'little')

            nt.write('\t'.join([str(idx), str(label), ':'.join(feat_vec)]) + '\n')

            idx += 1
            nbytes_feat += num_feats * 4
            nbytes_lab += 4

def convert_all(dataset):
    ## Input files
    graph_file = os.path.join(dataset, graph_file_name)
    features_file = os.path.join(dataset, features_file_name)
    labels_file = os.path.join(dataset, labels_file_name)

#    print("Converting edge list")
#    idx_set = convert_graph(graph_file)

#    print("Generating train/val/test masks")
#    generate_masks(idx_set)

    print("Converting node data")
    convert_node_data(features_file, labels_file)


if __name__ == "__main__":
    dataset = sys.argv[1]
    convert_all(dataset)
    print("Converting data for " + dataset + " finished")
