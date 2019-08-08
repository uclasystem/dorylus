#! /bin/python3

import struct


num_vertices = 4039
num_features = 6


with open("features.bsnap", mode="rb") as fbsnap, open("features", "w+") as ftext:
    read_data = fbsnap.read()
    
    num_feats = struct.unpack("I", read_data[:4])[0]
    assert(num_feats == num_features)

    fmt_str = num_features * "f"
    for i in range(num_vertices):
        idx_start = 4 + i * num_features * 4
        idx_end = idx_start + num_features * 4
        feat_values = struct.unpack(fmt_str, read_data[idx_start:idx_end])

        for val in feat_values:
            ftext.write(str(val) + ", ")
        ftext.write("\n")

