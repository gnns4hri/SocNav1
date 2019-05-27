import json
import numpy as np
import math

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

    groupStd = []
    groupNSamp = [] 
    for times in range(1, 7):
        the_list = [ x for x in huge_map.keys() if len(huge_map[x]) == times]
        print('Number of scenarios USED {} times: {}'.format(times, len(the_list)))
        if times == 1:
            continue
        stdDev = []
        nScenarios = len(the_list)
        for l in range(nScenarios):
           key = the_list[l]
           scores = [huge_map[key][i]['score'] for i in range(times)]
           avg = sum(scores)/times
           newStdDev = math.sqrt(sum([pow(scores[i]-avg,2) for i in range(times)])/times)
           stdDev.append(newStdDev)
        if len(the_list) > 2:
           meanStdDev = math.sqrt(sum([pow(stdDev[i],2) for i in range(nScenarios)])/nScenarios)
           groupStd.append(meanStdDev)
           groupNSamp.append(nScenarios*times)
           print("Mean standard deviation {:.2f}".format(meanStdDev*100))

    k=len(groupStd)
    pooledStdDev = math.sqrt(sum([pow(groupStd[i],2)*(groupNSamp[i]-1) for i in range(k)])/(sum([(groupNSamp[i]-1) for i in range(k)])-k))
    print("Pooled standard deviation {:.2f}".format(pooledStdDev*100))


