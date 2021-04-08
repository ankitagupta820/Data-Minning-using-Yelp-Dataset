import pyspark
import sys
import json
from operator import add


review_file = sys.argv[1]
output_file = sys.argv[2]
p_type = sys.argv[3]
num_p = int(sys.argv[4])
top_n = int(sys.argv[5])

result= dict()

sc=pyspark.SparkContext("local[*]", "Task3")
sc.setLogLevel("ERROR")

partitions = sc.textFile(review_file).map(lambda line: (json.loads(line)['business_id'], 1))

if p_type != "default":
    partitions = partitions.partitionBy(int(num_p), lambda key: hash(key))

result['n_partitions'] = partitions.getNumPartitions()
result['n_items'] = partitions.glom().map(len).collect()
result['result'] = partitions.reduceByKey(add).filter(lambda pair: pair[1]>top_n).collect()

with open(output_file, 'w+') as output_file:
    json.dump(result, output_file)
output_file.close()




