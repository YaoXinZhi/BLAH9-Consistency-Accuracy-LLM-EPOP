import json
import sys
import re
import traceback

from datetime import datetime
from collections import defaultdict

import argparse


def log(msg):
    now = datetime.now()
    sys.stderr.write(now.strftime('[%Y-%m-%d %H:%M:%S] ') + msg + '\n')


parser = argparse.ArgumentParser(
    prog='run'
)
parser.add_argument('-i', '--instruction', type=str, action='store')
parser.add_argument('-d', '--document', type=str, action='store')
parser.add_argument('-o', '--output', type=str, action='store')
parser.add_argument('-m', '--model', type=str, action='store')
parser.add_argument('-e', '--repeat', type=int, default=5)


args = parser.parse_args()

REPEATS = args.repeat

CODEBLOCK = re.compile('```json(.*)```', re.DOTALL)
def merge_dicts(dict_list):
    merged_dict = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            merged_dict[key].extend(value)  # 合并列表
    return dict(merged_dict)

def parse_json(rep):

    filename = f'output/repetition/{args.model}/{args.instruction}/{args.document}/{rep}.txt'
    # print(filename)
    with open(filename) as f:
        content = f.read().strip()
    m = CODEBLOCK.fullmatch(content)
    if m is None:
        jc = content
    else:
        jc = m.group(1)
    # j = json.loads(jc)

    jc = clean_json_string(jc)
    jc = jc.strip('`')
    # print(jc)
    j = json.loads(jc)

    # fixme：在新的格式中，会返回一个列表两个字典，entities和relationships
    if type(j) == list and len(j) == 1:
        return j[ 0 ]
    elif type(j) == list and len(j) == 2:
        j[ 0 ].update(j[ 1 ])
        return j[0]
    elif type(j) == list:
        j = merge_dicts(j)
        return j

    return j

def clean_json_string(json_str):
    return re.sub(r'^```json\s*', '', json_str)  # 删除开头的 ```json

def get_entities(j):
    for e in j['entities']:
        etype = e['type']
        assert type(etype) == str
        ename = e['name']
        assert type(ename) == str
        yield etype.lower(), ename.lower()


def get_relations(j):
    for r in j['relationships']:
        rsource = r['source']
        assert type(rsource) == str
        rtype = r['type']
        assert type(rtype) == str
        rtarget = r['target']
        assert type(rtarget) == str
        yield rtype.lower(), rsource.lower(), rtarget.lower()

def get_annotations(j):
    yield from get_entities(j)
    yield from get_relations(j)

ANN_MAP = {}
for rep in range(1, REPEATS +1):
    try:
        ann = set(get_annotations(parse_json(rep)))
    except Exception as e:
        # print("An error occurred:")
        # traceback.print_exc()
        ann = set()
    ANN_MAP[rep] = ann

# ANN_MAP = dict((rep, set(get_annotations(parse_json_new(rep)))) for rep in range(1, REPEATS + 1))

ALL_ANNOTATIONS = set.union(*ANN_MAP.values())

def get_row(a):
    result = [0, 0]
    for ranns in ANN_MAP.values():
        idx = int(a in ranns)
        result[idx] += 1
    return result


# from https://github.com/Shamya/FleissKappa/blob/master/fleiss.py
def checkInput(rate, n):
    """ 
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError 
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer" 
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"


def fleissKappa(rate,n):
    """ 
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category 
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters   
    @return fleiss' kappa
    """
    N = len(rate)
    k = len(rate[0])

    #mean of the extent to which raters agree for the ith subject 
    PA = sum([(sum([i**2 for i in row])- n) / (n * (n - 1)) for row in rate])/N

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j**2 for j in [sum([rows[i] for rows in rate])/(N*n) for i in range(k)]])

    try:
        return (PA - PE) / (1 - PE)
    except ZeroDivisionError:
        return 1.0

matrix = list(get_row(a) for a in ALL_ANNOTATIONS)

try:
    print(fleissKappa(matrix, REPEATS))
except Exception as e:
    # print("An error occurred:")
    # traceback.print_exc()
    print(0.0)
    # input()


