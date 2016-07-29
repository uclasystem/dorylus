#!/usr/bin/python
import sys
from sets import Set
from itertools import izip

if len(sys.argv) < 3:
  print("Dude! Invoke: " + sys.argv[0] + " <filename1> <filename2>")
  exit(0)

filename1 = sys.argv[1]
filename2 = sys.argv[2]

cmerge = 0
csplit = 0

fh = open(filename1)
with open(filename1) as fh1, open(filename2) as fh2:
  for l1, l2 in izip(fh1, fh2):
    tpl1 = l1.split()
    tpl2 = l2.split()

    if tpl1[0] != tpl2[0]:
      print("ERROR: Two lines don't talk about some vertex: <" + l1.strip() + "> <" + l2.strip() + ">")

    if int(tpl1[1]) > int(tpl2[1]):
      cmerge = cmerge + 1

    if int(tpl1[1]) < int(tpl2[1]):
      csplit = csplit + 1
      print("SPLIT: <" + l1.strip() + "> <" + l2.strip() + ">")

print("Merges: " + str(cmerge) + ", Splits: " + str(csplit))
