import sys
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from random import randint
import binascii
import json
import math


# Define Input Variables
port = int(sys.argv[1])
output_file = sys.argv[2]

# local variables
# port = 9999
# output_file="data/HW6/task2_output"
f= open(output_file, "w")
f.write("Time,Ground Truth,Estimation")
f.close()


# Define Hash Parameters
M = 450
no_hash_fn = 45
no_buckets = 9
rows_per_bucket = int(no_hash_fn / no_buckets)

# Define Hash functions
hash_fns = [[randint(1, 100), randint(1, 100)] for i in range(no_hash_fn)]

# for i in range(0, no_hash_fn):
# 	h = []
# 	for j in range(0,2):
# 		rand_no = randint(1, 100)
# 		h.append(rand_no)
# 	hash_fns.append(h)

def no_trailingZeros(mystr):
    return len(mystr) - len(mystr.rstrip('0'))


def FMA(time, rdd):

	global m
	global no_hash_fn
	global no_buckets
	global rows_per_group
	global hash_fns

	data = rdd.collect()
	ground_truth_distinct= set()

	hashes= []
	buckets= []

	for d in data:
		item = json.loads(d)
		city = item["city"]
		ground_truth_distinct.add(city)

		city_num = int(binascii.hexlify(city.encode('utf8')),16)

		hash_arr= []
		bin_array= []
		for i in hash_fns:
			hash = (i[0] * city_num + i[1]) % M
			bin_no = bin(hash)[2:]
			hash_arr.append(hash)
			bin_array.append(bin_no)

		hashes.append(hash_arr)
		buckets.append(bin_array)

	# Calculate max no. of zeros for all cities, for each hash fn.

	Approximate = []

	for hash in range(0, no_hash_fn):
		max_trailing = -1
		for bucket in range(0, len(buckets)):
			trailing = no_trailingZeros(buckets[bucket][hash])
			if(trailing > max_trailing):
				max_trailing = trailing
		Approximate.append(math.pow(2,max_trailing))


	# group hash functions and find average/ group
	per_bucket_avg = []
	for i in range(0, no_buckets):
		avg = 0
		for j in range(0, rows_per_bucket):
			index = i * rows_per_bucket + j
			avg += Approximate[index]

		avg = round(avg/rows_per_bucket)
		per_bucket_avg.append(avg)

	per_bucket_avg.sort()
	predicted_num_distinct = per_bucket_avg[int(no_buckets / 2)]

	f = open(output_file, "a")
	f.write("\n"+str(time)+","+str(len(ground_truth_distinct))+","+str(predicted_num_distinct))


sc = SparkContext("local[3]")
streaming_context = StreamingContext(sc, 5)
stream = streaming_context.socketTextStream("localhost", port).window(30, 10)
stream.foreachRDD(FMA)

streaming_context.start()
streaming_context.awaitTermination()
f.close()