import json
import numpy as np

from os import listdir
from os.path import isfile, join



def getHugeMap(file_names):
    ret = dict()
    names = []

    for filename in file_names:
        subj = filename.split('.')[0].split('/')[-1].split('_')[-1]
        names.append(subj)
        print('Reading', filename)
        for line in open(filename, 'r').readlines():
            st = json.loads(line)
            st['subject'] = subj
            st['score'] = 0.01 * st['score']
            if st['identifier'] not in ret.keys():
                ret[st['identifier']] = list()
            ret[st['identifier']].append(st)
    return ret, sorted(names)

def averageWithoutOutlier(vs):
    median = np.median(vs)
    vs2 = [ x for x in vs ]
    # vs2 = [ x for x in vs if abs(median-x) < 0.2 ]
    if len(vs2) < 2:
        return None
    return np.mean(vs2)

if __name__ == '__main__':
    # Get a list of the raw data files we will be using
    data_path = 'raw_data'
    filenames = [data_path+'/'+f for f in listdir(data_path) if isfile(join(data_path, f))]
    # Get map
    huge_map, names = getHugeMap(filenames)


    print("scn", end='\t')
    names.append('avg')
    for subject in names:
        print(subject, end='\t')
    print()

    scores = dict()

    scores['avg'] = []

    for identifier, samples in huge_map.items():
        avg = averageWithoutOutlier( [structure['score'] for structure in samples] )
        assessors = []
        if avg is not None:
            scores['avg'].append( avg )
            for structure in samples:
                try:
                    scores[structure['subject']].append(structure['score'])
                except KeyError:
                    scores[structure['subject']] = [structure['score']]
                assessors.append(structure['subject'])
            print(identifier, end='\t')
            for subject in names:
                if subject in assessors:
                    print("{:.2f}".format(scores[subject][-1]), end='\t')
                else:
                    print('', end='\t')
            print('')
