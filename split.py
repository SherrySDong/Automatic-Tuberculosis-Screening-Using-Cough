import random
import sys
random.seed(sys.argv[1])

GS = open('../data/code/preparegs.txt', 'r')
patients = {}
for line in GS:
    line = line.strip()
    table = line.split('\t')
    if table[3] in patients:
        pass
    else:
        patients[table[3]] = 1

allpid = patients.keys()
train = {}
for pid in allpid:
    rrr = random.random()
    if (rrr<1.8):
        train[pid] = 1

GS = open('../data/code/preparegs.txt', 'r')
TRAIN = open('train_gs', 'w')
TEST = open('test_gs', 'w')
for line in GS:
    line = line.strip()
    table = line.split('\t')
    if table[3] in train:
        TRAIN.write(line)
        TRAIN.write('\n')
    else:
        TEST.write(line)
        TEST.write('\n')
