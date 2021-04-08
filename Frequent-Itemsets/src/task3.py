from pyspark.mllib.fpm import FPGrowth
import pyspark
import time
import sys

# #Defined constants
USER_ID = "user_id"
BUSINESS_ID = "business_id"

#Program inputs
# filter = 100
# threshold = 60
# input_file = "data/HW2/user-business.csv"
# output_file = "data/HW2/task3_output.csv"
# task2_Numbers_file = "data/HW2/task2_numbers.csv"
# task2_Buckets_file = "data/HW2/task2_baskets.csv"

# #Initialize Spark Context
sc = pyspark.SparkContext("local[*]", "Task3")
sc.setLogLevel("ERROR")
start = time.time()

filter = int(sys.argv[1])
threshold = int(sys.argv[2])
input_file=sys.argv[3]
output_file= sys.argv[4]
task2_Numbers_file = "task2_numbers.csv"
task2_Buckets_file = "task2_buckets.csv"

input_text = sc.textFile(input_file)

#Define market-basket model
input_rdd = input_text.map(lambda l: l.split(',')).map(lambda l: (l[0], l[1])).distinct().map(lambda l: (l[0],[l[1]]))
model_RDD = input_rdd.reduceByKey(lambda a, b: a+b).persist().filter(lambda l: l[0] != USER_ID).map(lambda l: l[1]).filter(lambda x: len(x) > filter)

minSupport = threshold/model_RDD.count()

model = FPGrowth.train(model_RDD, minSupport= minSupport, numPartitions=10)
result = model.freqItemsets().collect()
task3_numbers = str(len(result))

#Read Frequent Buckets from task 2.
task2_buckets_text = sc.textFile(task2_Buckets_file)
task2_buckets = task2_buckets_text.map(lambda l: l.split(",")).map(lambda l: set(l)).collect()
task3_buckets = model.freqItemsets().map(lambda l: l[0]).map(lambda l: set(l)).collect()

#Find intersection of Task 2 and Task 3
intersect_count =0
for basket in task2_buckets:
    if basket in task3_buckets:
        intersect_count += 1

#Read Frequent Itemsets from task 2.
with open(task2_Numbers_file,"r") as task2_no_file:
    task2_numbers = str(task2_no_file.readline())

#Write Output
with open(output_file, "w") as task3_output:
    task3_output.write("Task2,"+task2_numbers+"\n")
    task3_output.write("Task3,"+task3_numbers+"\n")
    task3_output.write("Intersection,"+str(intersect_count))


#Print Duration of runtime
end = time.time()
print("Duration: ", (end-start))






