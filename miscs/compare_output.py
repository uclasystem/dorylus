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


current = ''
diff_list = list()
diff_set = set()
for l in range(len(lines1)):
    if (lines1[l] == '' and lines2[l] == ''):
        continue
    if (lines1[l][0] == "#" and lines2[l][0] == "#"):
        if (lines1[l] != lines2[l]):
            print("Mismatch on line " + str(l) + ":" + lines1[l] + " | " + lines2[l])
            exit()

        current = lines1[l]

    try:
        line_sum1 = sum_line(lines1[l])
        line_sum2 = sum_line(lines2[l])
    except:
        print("ERROR ON LINE NO", l)
        exit()

    if (line_sum1 - line_sum2 > threshold):
        if (current not in diff_set):
            diff_list.append(current)
            diff_set.add(current)

for section in diff_list:
    print(section)
