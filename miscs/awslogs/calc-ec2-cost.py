import subprocess as sp
import sys
from dateutil.parser import parse

logs_command = []

type_to_price = { "c5.xlarge"   :  0.17,
                  "c5.2xlarge"  :  0.34,
                  "r5a.xlarge"  :  0.226,
                  "t3.medium"   :  0.0416,
                  "c5n.xlarge"  :  0.216,
                  "g3s.xlarge"  :  0.75 }


def calc_ec2_cost(graph_type, n_graph, weight_type, n_weight, start, end):
    graph_price = type_to_price[graph_type]
    weight_price = type_to_price[weight_type]

    total_seconds = (end - start).total_seconds()

    total_cost = graph_price * total_seconds * n_graph / 3600 \
                + weight_price * total_seconds * n_weight / 3600

    return total_cost


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: python sum_lambda_time.py <graph-type> <num-graph-nodes> <weight-type> <num-weight-nodes> <start> <end>")
        exit()

    if sys.argv[1] not in type_to_price or sys.argv[3] not in type_to_price:
        print("Server type parameters must be in", list(type_to_price.keys()))
        exit()

    graph_type = sys.argv[1]
    n_graph = int(sys.argv[2])
    weight_type = sys.argv[3]
    n_weight = int(sys.argv[4])

    startdt = parse(sys.argv[5])
    enddt = parse(sys.argv[6])

    ec2_cost = calc_ec2_cost(graph_type, n_graph, weight_type, n_weight, startdt, enddt)

    print(ec2_cost)
