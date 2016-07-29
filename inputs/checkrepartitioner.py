#!/usr/bin/python

import sys

if len(sys.argv) < 3:
    print("Dude! Invoke: " + sys.argv[0] + " <parts-file> <reparts-file>")
    exit(-1)

partsFile = sys.argv[1]
repartsFile = sys.argv[2]

fh = open(partsFile)
parts = []
for line in fh:
    parts.append(int(line.strip()))

fh.close()

fh = open(repartsFile)
i = 0
for line in fh:
    if int(line.strip()) == parts[i]:
        print("This is wrong. Line " + str(i) + " has same value " + str(parts[i]))

    i = i + 1

fh.close()
