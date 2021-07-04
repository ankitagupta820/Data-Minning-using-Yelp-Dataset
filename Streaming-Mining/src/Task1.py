from pyspark import SparkContext, SparkConf
from binascii import hexlify
import sys
import time
import random
import csv
import json


def createHashFunctions(no_functions, hash_buckets):

    hashFunctions = list()

    def gen_func(A, B, M):
        def hash(X):
            return ((A * X + B) % 233333333333) % M
        return hash

    A = random.sample(range(1, sys.maxsize - 1), no_functions)
    B = random.sample(range(0, sys.maxsize - 1), no_functions)

    for a_param, b_param in zip(A, B):
        hashFunctions.append(gen_func(a_param, b_param, hash_buckets))

    return hashFunctions


def predict(city, bit_array):

    if city is not None and city != "":
        city_num = int(hexlify(city.encode('utf8')), 16)
        sig = set([func(city_num) for func in Hash_Functions])
        yield 1 if sig.issubset(bit_array) else 0
    else:
        yield 0


def writeToOutput(output, file):

    with open(file, "w+", newline="") as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(output)

NO_HASH_FUNC = 7
BIT_ARR_LEN = 7000

tick = time.time()

# Define input paths
# train_business = "data/HW6/business_first.json"
# test_business = "data/HW6/business_second.json"
# output_filepath= "data/HW6/task1_output.csv"

train_business = sys.argv[1]
test_business = sys.argv[2]
output_filepath = sys.argv[3]

# Spark settings
conf = SparkConf().setMaster("local[3]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

# Read the training json file
train_RDD = sc.textFile(train_business).map(lambda line: json.loads(line)).map(lambda x: x["city"]).distinct()\
	.filter(lambda city: city != "")\
	.map(lambda city: int(hexlify(city.encode('utf8')), 16))

# Create hash functions
Hash_Functions = createHashFunctions(NO_HASH_FUNC, BIT_ARR_LEN)

# Hash cities of train data
hashed_cities = train_RDD.flatMap(lambda i: [hash(i) for hash in Hash_Functions]).collect()
print(hashed_cities[:1])

# Define bit array
bit_array = set(hashed_cities)

# Predict on test data
result_rdd = sc.textFile(test_business).map(lambda line: json.loads(line)).map(lambda x: x["city"]).flatMap(lambda city: predict(city, bit_array))

# Write to Output File
writeToOutput(result_rdd.collect(), output_filepath)
tock = time.time()
print("Duration:",(tock - tick))