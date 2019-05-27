import sys
import json
import random

from os import listdir
from os.path import isfile, join

# Get a list of the raw data files we will be using
data_path = sys.argv[1]
filenames = [data_path+'/'+f for f in listdir(data_path) if isfile(join(data_path, f))]
print(filenames)

# Definition of how will we split
splits = [ ('socnav_dev.json', 0.06), ('socnav_test.json', 0.06), ('socnav_training.json', 0)]

# Read all the samples from all raw data files
lines = []
id_set = set()
for filename in filenames:
    a = len(lines)
    for line in open(filename, 'r').readlines():
        lines.append(line.strip())
        id_set.add(json.loads(line)['identifier'])
    b = len(lines)
    print (filename, b-a)
# Shuffle the data before splitting
random.shuffle(lines)

# Some output...
N = len(lines)
print('Total:', N, 'samples')
print('Total DIFFERENT scenarios:', len(id_set))

# Prepare for data augmentation
to_duplicate = []

# Main loop. Do the splitting and prepare for data augmentation *of the training set*
for split in splits:
    f = open(split[0], 'w')
    if split[1] == 0:
        n = len(lines)
    else:
        n = int(split[1]*N)
    print ('Writing to ', split[0], n, 'lines')
    for i in range(n):
        line_text = lines.pop()+'\n'
        if split[1] == 0:
            to_duplicate.append(line_text)
        f.write(line_text)
    f.close()

# Data augmentation
dups = []
for line in to_duplicate:
    dups.append(line.strip())
    structure = json.loads(line)
    structure['identifier'] = structure['identifier'][:-1] + 'B'

    for i in range(len(structure['humans'])):
        structure['humans'][i]['xPos']        *= -1.
        structure['humans'][i]['orientation'] *= -1.

    for i in range(len(structure['objects'])):
        structure['objects'][i]['xPos']        *= -1.
        structure['objects'][i]['orientation'] *= -1.

    for i in range(len(structure['room'])):
        structure['room'][i][0] *= -1.
    dups.append(json.dumps(structure))

random.shuffle(dups)
f = open('socnav_training_dup.json', 'w')
for line in dups:
    f.write(line+'\n')
f.close()
