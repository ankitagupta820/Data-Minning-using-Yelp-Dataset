from pyspark import SparkContext
from collections import defaultdict
from itertools import combinations
import time
import math
import json
import sys
import random

# Unique Business: 10,253
# No of combinations: 52,556,878
# Business pairs with more than 3 co-rated users: 1,171,857
# Business pairs with sim < 0 : 614162

sc = SparkContext()
tickTime = time.time()

# constants
BUSINESS_ID = "business_id"
USER_ID = "user_id"
FEATURES = "features"
STARS = "stars"
ITEM_BASED = "item_based"
USER_BASED = "user_based"

primes = [(2, 503), (3, 509), (5, 521), (7, 523), (11, 541), (13, 547), (17, 557), (19, 563), (23, 569), (29, 571),
          (31, 577), (37, 587), (39, 587), (41, 593), (43, 593), (47, 599), (53, 601), (59, 607), (61, 613), (67, 617),
          (71, 619), (73, 631), (79, 641), (83, 643), (89, 647), (97, 653), (101, 659), (103, 661), (107, 673), (109, 677),
          (113, 683), (127, 691), (131, 701), (137, 709), (139, 719), (149, 727), (151, 733), (157, 739), (163, 743), (167, 751)]
         # (173, 757), (179, 761), (181, 769), (191, 773), (193, 787), (197, 797), (199, 809), (211, 811), (223, 821), (227, 821)]

# primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 39, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,227]
n_hash_functions = len(primes)
r = 1
b = int(n_hash_functions / r)

# local run files
input_file = "data/HW3/train_review.json"
model_file = "data/HW3/task3_output.json"
CF_type = USER_BASED


# input_file, model_file, CF_type = sys.argv[1:]


def pearson(users1, users2):

    common_users = set(users1.keys()) & set(users2.keys())

    u1_ratings = [users1[common] for common in common_users]
    u1_avg = sum(u1_ratings) / len(u1_ratings)
    u1_norm_ratings = [r - u1_avg for r in u1_ratings]

    u2_ratings = [users2[common] for common in common_users]
    u2_avg = sum(u2_ratings) / len(u2_ratings)
    u2_norm_ratings = [r - u2_avg for r in u2_ratings]

    num = 0
    for r in range(len(u1_norm_ratings)):
        t = u1_norm_ratings[r] * u2_norm_ratings[r]
        num += t

    denom_u1 = math.sqrt(sum([r ** 2 for r in u1_norm_ratings]))
    denom_u2 = math.sqrt(sum([r ** 2 for r in u2_norm_ratings]))

    denom = denom_u1 * denom_u2

    if num > 0 and denom > 0:
        sim = num / denom
    else:
        sim = 0

    return sim


def minHash(x):
    # global rows
    # global primes
    signatures = [min(((p[0] * row_no + p[1]) % 10343) % rows for row_no in x[1]) for p in primes]
    return (x[0], signatures)  # business - [hash signatures .. 20 in nos]


def get_signature_bands(x):
    business_id = x[0]
    signatures = x[1]

    bands = []
    rowindex = 0

    for band_no in range(0, b):
        band = []
        for row in range(0, r):
            band.append(signatures[rowindex])
            rowindex = rowindex + 1
        bands.append(((band_no, tuple(band)), [business_id]))
        band.clear()

    return bands


def get_candidate_pairs(x):
    pair_list = set()
    b_list = x[1]
    #print(tuple(combinations(b_list,2)))
    #b_list.sort()

    for c in combinations(b_list,2):
        l= tuple(sorted(c))
        pair_list.add(l)
    # for i in range(0, len(b_list)):
    #     for j in range(i + 1, len(b_list)):
    # #         pair_list.append(((b_list[i], b_list[j]), 1))
    print("PAIR LIST: ", pair_list)
    return pair_list


def find_jaccard_similarity(x):
    b1 = x[0][0]
    b2 = x[0][1]

    users_b1 = set(user_business_rdd[b1])
    users_b2 = set(user_business_rdd[b2])

    intersection = len(users_b1 & users_b2)
    union = len(users_b1 | users_b2)

    if (intersection < 3):
        return ((b1, b2), 0)
    else:
        return ((b1, b2), float(intersection / union))


input_RDD = sc.textFile(input_file).map(lambda line: json.loads(line)).map(
    lambda line: (line[USER_ID], line[BUSINESS_ID], line[STARS]))
businesses = input_RDD.map(lambda l: l[1]).distinct().collect()

if CF_type == ITEM_BASED:

    # collect (business_id, [(user_id,rating)]) for all b rated more than 3 times
    B_UR = input_RDD.map(lambda l: (l[1], [(l[0], l[2])])).reduceByKey(lambda a, b: a + b).filter(
        lambda l: len(l[1]) >= 3).collect()

    # {business_id: {u1:r1, u2:r2}}
    B_UR_map = defaultdict(dict)
    for list in B_UR:
        business_id = list[0]
        ratings = list[1]
        for rating in ratings:
            B_UR_map[business_id][rating[0]] = rating[1]

    # business pairs with 3 or more corated users similarity > 0
    with open(model_file, 'w+') as f:

        for pair in combinations(businesses, 2):
            business1 = pair[0]
            business2 = pair[1]

            users1 = B_UR_map[business1]
            users2 = B_UR_map[business2]

            common_users = set(users1.keys()) & set(users2.keys())
            if len(common_users) >= 3:
                sim = pearson(users1, users2, common_users)
                if sim > 0:
                    f.write(json.dumps({"b1": business1, "b2": business2, "sim": sim}) + "\n")

        tockTime = time.time()
        print("Duration: ", tockTime - tickTime)

else:
    tick = time.time()
    # (user_id, (business_id, rating))
    U_BR = input_RDD.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda a, b: a + b).filter(
        lambda l: len(l[1]) >= 3).collect()

    # {user_id: {b1:r1, b2:r2}}
    U_BR_map = defaultdict(dict)
    for list in U_BR:
        user_id = list[0]
        ratings = list[1]
        for rating in ratings:
            U_BR_map[user_id][rating[0]] = rating[1]

    # --------LSH MIN HASH------------
    # user_business_rdd = input_RDD.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y).collectAsMap()
    #
    # allUsers = input_RDD.map(lambda kv: (kv[0], 1)).keys().distinct()
    # allBusiness = input_RDD.map(lambda kv: (kv[1], 1)).keys().distinct()
    #
    # rows = allBusiness.count()
    # cols = allUsers.count()
    #
    # users = allUsers.collect()
    # print("Unique Business:", len(users))
    #
    # business = allBusiness.collect()
    # print("Unique Users:", len(business))
    #
    # # Create index for users and businesses
    # business_index = {}
    # for B in range(0, rows):
    #     business_index[business[B]] = B
    #
    # # Create characteristic matrix i.e user_id - [business_id indices]
    # char_matrix = input_RDD.map(lambda a: (a[0], [business_index[a[1]]])).reduceByKey(lambda a, b: a + b)
    #
    # # create signature matrix business_id - [50 signatures]
    # buckets = rows / n_hash_functions
    # sig_matrix = char_matrix.map(lambda business: minHash(business))
    # print("Minhash complete")
    #
    # # Perform LSH
    # sig_bands_matrix = sig_matrix.flatMap(lambda x: get_signature_bands(x))
    #
    # agg_candidates = sig_bands_matrix.reduceByKey(lambda a, b: a + b).filter(lambda business: len(business[1]) > 1)
    # print(agg_candidates.take(1))
    # candidate_pairs = agg_candidates.flatMap(lambda b: get_candidate_pairs(b)).distinct().collect()
    # print("Candidate Pairs Generated")


    #map {business_id: business_index}
    business_map = dict()
    for b in range(len(businesses)):
        business_map[businesses[b]] = b

    #(business_id, user_id)
    input_index_RDD = input_RDD.map(lambda kv: (business_map[kv[1]], kv[0]))

    review_index_rdd = input_RDD.map(lambda x: (business_map[x[1]], x[0]))

    n = 50;
    band = 50;
    row = 1
    hashed_b = defaultdict(list)
    for j in range(len(businesses)):
        for k in range(n):
            a = random.randint(1, 100000)
            b = random.randint(1, 100000)
            p = 10343
            m = 10253
            hashed_b[j].append(((a * j + b) % p) % m)

    signature_matrix = review_index_rdd.groupByKey() \
        .map(lambda x: (x[0], list(set(x[1])))) \
        .map(lambda x: (x[1], hashed_b[x[0]])) \
        .map(lambda x: ((user_id, x[1]) for user_id in x[0])) \
        .flatMap(lambda x: x) \
        .groupByKey().map(lambda x: (x[0], [list(x) for x in x[1]])) \
        .map(lambda x: (x[0], [min(col) for col in zip(*x[1])])).collect()

    # generate candidate pairs
    candidates = set()

    for band_num in range(band):
        bucket = defaultdict(set)
        for signature in signature_matrix:
            start_index = band_num * row
            value = tuple()
            for row_num in range(row):
                value += (signature[1][start_index + row_num],)
            hashed_value = hash(value)
            bucket[hashed_value].add(signature[0])
            # print(bucket)
        for li in bucket.values():
            if len(li) >= 2:
                for pair in combinations(li, 2):
                    candidates.add(tuple(sorted(pair)))

    with open(model_file, 'w+') as f:
        for pair in candidate_pairs:
            u1 = pair[0][0]
            u2 = pair[1][1]
            business1 = U_BR_map[u1]
            business2 = U_BR_map[u2]

            common_business = set(business1.keys()) & set(business2.keys())
            if len(common_business) >= 3:
                pearsonSimilarity = pearson(business2, business2)
                if pearsonSimilarity > 0:
                    f.write(json.dumps({'u1': u1, 'u2': u2, 'sim': pearsonSimilarity}) + '\n')
                # if pearsonSimilarity > 0:
                #     if find_jaccard_similarity(business1, business2) >= 0.01:


    # # perform Jaccard similarity on candidate pairs
    # JS_rdd = candidate_pairs.map(lambda x: find_jaccard_similarity(x)).filter(lambda pair: pair[1] >= 0.01)
    # # result_rdd = JS_rdd.map(lambda r: (r[0][1], (r[0][0], r[1]))).sortByKey().map(
    # #     lambda r: (r[1][0], (r[0], r[1][1]))).sortByKey()
    # print(JS_rdd.take(10))
    # print("Jaccard Sim")
    #
    # print(JS_rdd.take(10))
    tock = time.time()
    print("Duration:", tock - tick)


    #
    # # (business_index, [user_id, user_id, ..])
    # sig1 = input_index_RDD.map(lambda kv: (kv[0], [kv[1]])).reduceByKey(lambda a,b : a+b).map(lambda kv: (kv[0], list(set(kv[1]))))
    #
    # # ([u1, u2,..], business_hash)
    # sig2 = sig1.map(lambda x: (x[1], hashed_business[x[0]]))
    #
    # # (user_id, business_hash)
    # sig3 = sig2.map(lambda x: ((user_id, x[1]) for user_id in x[0]))
    # sig4 = sig3.flatMap(lambda x: x)
    #
    #     .groupByKey().map(lambda x: (x[0], [list(x) for x in x[1]])) \
    #     .map(lambda x: (x[0], [min(col) for col in zip(*x[1])])).collect()
    #
    # # generate signature matrix in data structure of nested list: 26184 x n
    # # [(user_id, [h1, h2, ... , hn])]
    #
    #
    #
    # review_index_rdd = input_RDD.map(lambda x: (business_map[x[1]], x[0]))

    print("User-Based CF")
