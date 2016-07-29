#!/usr/bin/python
import sys
from sets import Set

def commSimilarity(set1, set2):
  same = set1.intersection(set2)
  return ((len(same) * 1.0) / max(len(set1), len(set2)), len(set2) - len(same))


if len(sys.argv) < 3:
  print("Dude! Invoke: " + sys.argv[0] + " <filename1> <filename2>")
  exit(0)

filename1 = sys.argv[1]
filename2 = sys.argv[2]

comm1 = {}

fh = open(filename1)
for line in fh:
  tpl = line.split()
  
  c = int(tpl[1])
  v = int(tpl[0])

  if c in comm1:
    comm1[c].add(v)
  else:
    comm1[c] = Set([v])

fh.close()

comm2 = {}

fh = open(filename2)
for line in fh:
  tpl = line.split()
  
  c = int(tpl[1])
  v = int(tpl[0])

  if c in comm2:
    comm2[c].add(v)
  else:
    comm2[c] = Set([v])

fh.close()

print(str(len(comm1)) + " " + str(len(comm2)))

diff = 0
for c1 in comm1:
  for c2 in comm2:
    (simm, d) = commSimilarity(comm1[c1], comm2[c2])
    if simm >= 0.1:
      if simm < 1.0:
        print("Simmilarity(" + str(c1) + ", " + str(c2) + ") = " + str(simm) + " difference = " + str(d))
        diff = diff + d
      else:
        print("Simmilarity(" + str(c1) + ", " + str(c2) + ") = " + str(simm))

print("Total " + str(diff) + " vertices are incorrect")
