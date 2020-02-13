#! /usr/bin/python3

import sys

if (len(sys.argv) < 3):
	print("Need two file names")
	exit()

filename1 = sys.argv[1]
filename2 = sys.argv[2]

f1 = open(filename1, 'r')
f2 = open(filename2, 'r')

lines1 = f1.read().split('\n')
lines2 = f2.read().split('\n')


if (len(lines1) != len(lines2)):
	print("File line counts don't match")
	exit()

threshold = .0001

def sum_line(line):
	total = 0
	for f in line.split():
		total += float(f)

	return total

epoch = -1
layer = -1

for l in range(len(lines1)):
	if (lines1[l] == '' and lines2[l] == ''):
		continue
	if (lines1[l][0] == "#" and lines2[l][0] == "#"):
		continue

	try:
		line_sum1 = sum_line(lines1[l])
		line_sum2 = sum_line(lines2[l])
	except:
		print("ERROR ON LINE NO", l)
		exit()

	if (line_sum1 - line_sum2 > threshold):
		print("DIFF ON LINE " + str(l) + " EXCEEDS THRESHHOLD")
		print(lines1[l])
		print(lines2[l])
