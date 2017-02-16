#!/usr/bin/python

import sys
from subprocess import *


if len(sys.argv) < 4:
  print("Dude! Invoke: " + sys.argv[0] + " <syncdirname> <ipfilename> <dstdirname>")
  exit(0)

sdirname = sys.argv[1].strip()
filename = sys.argv[2].strip()
ddirname = sys.argv[3].strip()

sinputs = sdirname + "/realinputs/";
dinputs = "/mnt0/inputs";

fh = open(filename)
for line in fh:
  line.strip()

  p = Popen(["rsync", "-av", "--exclude", ".git", "--exclude", "build", "--exclude", "realinputs", "--exclude", "run", sdirname, line.strip() + ":" + ddirname])
  p.communicate()

  p = Popen(["rsync", "-av", sinputs, line.strip() + ":" + dinputs])
  p.communicate()

fh.close()
