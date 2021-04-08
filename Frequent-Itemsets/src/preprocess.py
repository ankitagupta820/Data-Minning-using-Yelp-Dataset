import pyspark
import json
import csv
import time

# Initialize Spark Context
sc = pyspark.SparkContext("local[*]", "Task1")
sc.setLogLevel("ERROR")

# Input data
business_file = "data/business_new.json"
review_file = "data/review_new.json"
output_file = "data/HW2/user-business.csv"

# define constants
State = "NV"
BUSINESS_ID = "business_id"
USER_ID = "user_id"
STATE = "state"

start = time.time()

# list of businesses operating in Nevada
business_state_RDD = sc.textFile(business_file)\
    .map(lambda line: json.loads(line))\
    .map(lambda business: (business[BUSINESS_ID], business[STATE]))

b_list = business_state_RDD\
    .filter(lambda business: business[1] == State)\
    .map(lambda business: business[0]).collect()

# list of (review_id, business_id) where business_id is in blist
review_business_RDD = sc.textFile(review_file)\
    .map(lambda line: json.loads(line))\
    .map(lambda review: (review[USER_ID], review[BUSINESS_ID]))\
    .filter(lambda review: review[1] in b_list)\
    .collect()


with open(output_file,"w+", newline='') as ofile:
    csvwriter = csv.writer(ofile)
    csvwriter.writerow(["user_id", "business_id"])
    for row in review_business_RDD:
        csvwriter.writerow(row)

end = time.time()
print("duration:", end-start)






