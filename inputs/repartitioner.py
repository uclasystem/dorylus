#!/usr/bin/python

import sys

if len(sys.argv) < 3:
  print("Dude! Invoke: " + sys.argv[0] + " <parts-file> <num-partitions>")
  exit(-1)

partsFile = sys.argv[1]
numParts = int(sys.argv[2])

fh = open(partsFile)

for line in fh:
  p = int(line.strip())
  print(str(p % numParts))

fh.close()

