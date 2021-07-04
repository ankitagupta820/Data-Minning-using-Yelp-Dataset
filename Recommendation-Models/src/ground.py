from pyspark import SparkContext
import json
import itertools

input_file = "data/HW3/train_review.json"
output_file = "data/HW3/ground.json"

BUSINESS_ID = "business_id"
USER_ID = "user_id"

sc = SparkContext()

text_data= sc.textFile(input_file)
json_data= text_data.map(lambda l: json.loads(l))
business_user_rdd = json_data\
    .map(lambda review: (review[BUSINESS_ID], [review[USER_ID]]))\
    .reduceByKey(lambda x, y: x+y)\
    .collectAsMap()

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
char_matrix = json_data\
    .map(lambda a: (a[BUSINESS_ID], [users_index[a[USER_ID]]]))\
    .reduceByKey(lambda a, b: a+b)\
    .collectAsMap()

combinations = itertools.combinations(business,2)
result = []

for c in combinations:

    items = list(c)
    items.sort()
    u1 = set(char_matrix[items[0]])
    u2 = set(char_matrix[items[1]])
    intersection = u1 & u2
    union = u1 | u2

    jaccardd_sim = len(intersection)/len(union)
    if(jaccardd_sim>=0.05):
        result.append([items[0], items[1], jaccardd_sim])

#Write to output file
with open(output_file,'w') as f:
    for pair in result:
        json.dump({"b1":pair[0], "b2": pair[1], "sim": pair[2]}, f)
        f.write("\n")
f.close()

sort_data= sc.textFile(output_file)
s_data= sort_data.map(lambda l: json.loads(l))\
    .map(lambda l: (l["b2"], (l["b1"], l["sim"])))\
    .sortByKey()\
    .map(lambda r: (r[1][0], (r[0], r[1][1])))\
    .sortByKey()

with open(output_file,'w') as f:
    for pair in s_data.collect():
        json.dump({"b1": pair[0], "b2": pair[1][0], "sim": pair[1][1]}, f)
        f.write("\n")
f.close()

