import sys
import re

usage = '''
Usage: python3 {} <weightserver-out file> <dataset> <mode>
    <weightserver-out file>: path and filename of weightserver output file.
    <dataset>: [amazon | reddit-small | reddit-large]
    <mode>: [sync | s=0 | s=1]
'''

keyword = '[ WS   0 ] Epoch'
def main():
    if len(sys.argv) < 4:
        print(usage.format(sys.argv[0]))
        exit(-1)

    wsout = sys.argv[1]
    dataset = sys.argv[2]
    mode = sys.argv[3]

    accs = []
    with open(wsout, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l[:len(keyword)] == keyword:
                # parse the acc data from lines like "[ WS   0 ] Epoch 1, acc: 0.0149, loss: 3.8260"
                acc = float(re.findall("\d+\.\d+", l)[0])
                accs.append(acc)

    print(accs)
    acc_filename = dataset + '-acc-' + mode + '.txt'
    with open(acc_filename, 'w') as f:
        for acc in accs:
            f.write('{:.4f}\n'.format(acc))

main()
