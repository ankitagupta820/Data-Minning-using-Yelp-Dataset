import sys
from pyspark import SparkContext
import time
import json
import collections
import copy
import math
import time
from functools import reduce
from itertools import combinations
from operator import add


tickTime = time.time()
sc = SparkContext()

# #FilePaths
# input_file, output_file = sys.argv[1:]

#LocalRun
input_file = "data/HW3/train_review.json"
output_file = "data/HW3/task1_output.json"

#Constants
BUSINESS_ID = "business_id"
USER_ID = "user_id"
primes = [1, 3, 9, 11, 13, 17, 19, 27, 29, 31, 33, 37, 39, 41, 43, 47, 51, 53, 57, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,227]
n_hash_functions = len(primes)
r = 1
b = int(n_hash_functions/ r)

def minHash(x):
    # global rows
    # global primes
    signatures = [min((p*row_no + 1) % rows for row_no in x[1]) for p in primes]
    return (x[0], signatures) #business - [hash signatures .. 20 in nos]

def get_signature_bands(x):

    business_id = x[0]
    signatures = x[1]

    bands = []
    rowindex = 0

    for band_no in range(0, b):
        band = []
        for row in range(0, r):
            band.append(signatures[rowindex])
            rowindex = rowindex+1
        bands.append(((band_no, tuple(band)), [business_id]))
        band.clear()

    return bands

def get_candidate_pairs(x):

    pair_list = []
    b_list = x[1]
    b_list.sort()
    for i in range(0, len(b_list)):
        for j in range(i+1, len(b_list)):
            pair_list.append(((b_list[i], b_list[j]), 1))

    return pair_list

def find_jaccard_similarity(x):
    b1 = x[0][0]
    b2 = x[0][1]

    users_b1 = set(business_user_rdd[b1])
    users_b2 = set(business_user_rdd[b2])

    jaccard_sim = float(len(users_b1 & users_b2) / len(users_b1 | users_b2))
    return ((b1,b2), jaccard_sim )

#Read input data
text_data= sc.textFile(input_file)
json_data= text_data.map(lambda l: json.loads(l))
business_user_rdd = json_data.map(lambda review: (review[BUSINESS_ID], [review[USER_ID]])).reduceByKey(lambda x, y: x+y).collectAsMap()

allUsers = json_data.map(lambda review: review[USER_ID]).distinct()
allBusiness = json_data.map(lambda review: review[BUSINESS_ID]).distinct()

rows = allUsers.count()
cols = allBusiness.count()

users = allUsers.collect()
business = allBusiness.collect()

# Create index for users and businesses
users_index = {}
for U in range(0, rows):
    users_index[users[U]] = U

business_index ={}
for B in range(0, cols):
    business_index[business[B]] = B

#Create characteristic matrix i.e business_id - [user_id indices]
char_matrix = json_data.map(lambda a: (a[BUSINESS_ID], [users_index[a[USER_ID]]])).reduceByKey(lambda a, b: a+b)

#create signature matrix business_id - [20 signatures]
buckets = rows/n_hash_functions
sig_matrix = char_matrix.map(lambda business: minHash(business))


#Perform LSH
sig_bands_matrix = sig_matrix.flatMap(lambda x: get_signature_bands(x))
agg_candidates = sig_bands_matrix.reduceByKey(lambda a, b: a+b).filter(lambda business: len(business[1]) > 1)
candidate_pairs = agg_candidates.flatMap(lambda b: get_candidate_pairs(b)).distinct()

# perform Jaccard similarity on candidate pairs
JS_rdd = candidate_pairs.map(lambda x: find_jaccard_similarity(x)).filter(lambda pair: pair[1] >= 0.05)
result_rdd = JS_rdd.map(lambda r: (r[0][1], (r[0][0], r[1]))).sortByKey().map(lambda r: (r[1][0], (r[0], r[1][1]))).sortByKey()

#Write to output file
with open(output_file,'w') as f:
    for pair in result_rdd.collect():
        json.dump({"b1":pair[0], "b2": pair[1][0], "sim": pair[1][1]}, f)
        f.write("\n")
f.close()

tockTime = time.time()
print("Duration: ",tockTime-tickTime)


#Check Precision and recall
ground = sc.textFile("data/HW3/ground.json").map(lambda l: json.loads(l)).map(lambda x: (x["b1"], x["b2"]))
Ground_truth = list(ground.collect())

output = JS_rdd.map(lambda x: (x[0][0], x[0][1]))
Output_pairs = list(output.collect())

TP= list(output.intersection(ground).collect())

precision = len(TP) / len(Output_pairs)
recall = len(TP) / len(Ground_truth)
print("Precision:", precision)
print("Recall: ", recall)