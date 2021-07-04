from pyspark import SparkContext
import sys
import time
import json
import math

def calculate_sim(u_profile, b_profile):

    if u_profile == None or b_profile == None:
        similarity = 0.0
    else:
        u_set = set(u_profile)
        b_set = set(b_profile)
        similarity = len(u_set & b_set) / (math.sqrt(len(u_set)) * math.sqrt(len(b_set)))

    return similarity

sc = SparkContext()
tickTime = time.time()

#Constants
BUSINESS_ID = "business_id"
USER_ID = "user_id"
FEATURES="features"
SIMILARITY="sim"

#local run
# test_file ="data/HW3/test_review.json"
# model_file="data/HW3/task2_output.json"
# output_file ="data/HW3/task2_predictions.json"


# sys args
test_file, model_file, output_file = sys.argv[1:]

# read data into RDDs
test_RDD = sc.textFile(test_file).map(lambda line: json.loads(line)).map(lambda line: (line[USER_ID], line[BUSINESS_ID]))
model_RDD = sc.textFile(model_file).map(lambda line: json.loads(line))

b_profile = model_RDD.filter(lambda line: BUSINESS_ID in line).map(lambda line: (line[BUSINESS_ID], line[FEATURES]))
u_profile = model_RDD.filter(lambda line: USER_ID in line).map(lambda line: (line[USER_ID], line[FEATURES]))

# convert business profile to map
b_profile_map = dict()
u_profile_map = dict()

for business in b_profile.collect():
    b_profile_map[business[0]] = business[1]

for user in u_profile.collect():
    u_profile_map[user[0]] = user[1]

#calculate cosine similarity
predict = test_RDD.map(lambda l: (l[0], l[1])).map(lambda l: ((l[0], l[1]), calculate_sim(u_profile_map.get(l[0]), b_profile_map.get(l[1])))).filter(lambda l: l[1] >= 0.01).collect()

#write results to file
with open(output_file, 'w+') as f:
	for item in predict:
		f.write(json.dumps({USER_ID: item[0][0], BUSINESS_ID: item[0][1], SIMILARITY: item[1]}) + '\n')

tockTime = time.time()
print("Duration: ", tockTime-tickTime)
