import sys
import time
import itertools
import os
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from operator import  add
from graphframes import *


# os.environ["PYSPARK_SUBMIT_ARGS"] = (
# "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")
conf = SparkConf()\
    .setMaster("local[3]")\
    .set("spark.executor.memory", "4g")\
    .set("spark.driver.memory", "4g")


sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)
spark = SparkSession(sc)
hasattr(sc, "toDF")
tick = time.time()

#local run
threshold = 7
input_file = "data/HW4/ub_sample_data.csv"
output_file = "data/HW4/task1_output.csv"

# threshold = int(sys.argv[1])
# input_file = sys.argv[2]
# output_file = sys.argv[3]

def getEdges(pair):

    edges = []
    if len(set(UB_map[pair[0]]).intersection(set(UB_map[pair[1]]))) >= threshold:
        edges.append(tuple(pair))
        edges.append(tuple((pair[1], pair[0])))

    return edges

# read the original json file and remove the header
input_rdd = sc.textFile(input_file)

h = input_rdd.first()

# create map {user_id: [Business_id...]}
UB = input_rdd.filter(lambda row: row != h)\
    .map(lambda l: (l.split(',')[0], l.split(',')[1]))\
    .groupByKey()\
    .mapValues(lambda business: sorted(list(business)))

UB_map = UB.collectAsMap()

# create user pairs
user_pair_list = list(itertools.combinations(list(UB_map.keys()), 2))
user_pair_rdd = sc.parallelize(user_pair_list)


Edges = user_pair_rdd.map(lambda pair: getEdges(pair)).reduce(add)

Edges_rdd = sc.parallelize(Edges)
vertices = list(set(Edges_rdd.map(lambda x: x[0]).collect() + Edges_rdd.map(lambda x: x[1]).collect()))

V_df = sqlCtx.createDataFrame(sc.parallelize(vertices).map(lambda user: (user,)), ['id'])
E_df = sqlCtx.createDataFrame(Edges_rdd,["src", "dst"])
#
#create Graph and run algorithm
G = GraphFrame(V_df, E_df)
communities = G.labelPropagation(maxIter=5)


#communities.select("id", "label").show()
communities_rdd = communities.rdd.coalesce(1)\
    .map(lambda row: (row[1], row[0]))\
    .groupByKey()\
    .map(lambda row: sorted(list(row[1])))\
    .sortBy(lambda users: (len(users), users))


#Write output to file
with open(output_file, 'w+') as f:
    for c in communities_rdd.collect():
        f.writelines((str(c)[1:-1]) + "\n")
    f.close()

tock = time.time()
print("Task 1 Duration: ",(tock-tick))
