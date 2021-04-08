import pyspark
import sys
import json
import re
import heapq


review_file="data/review.json"
business_file="data/business.json"
output_file = "data/task2_ans"
is_spark = "no_spark"
top_n = 10

result= dict()


B_ID="business_id"
CAT = "categories"
STARS="stars"

# review_file = sys.argv[1]
# business_file = sys.argv[2]
# output_file = sys.argv[3]
# is_spark = sys.argv[4]
# top_n = int(sys.argv[5])

def average_star_rating(reviewRDD, businessRDD, n):

    category_rdd = businessRDD.filter(lambda b: (b[1] is not "") & (b[1] is not None)).mapValues(lambda b: [b.strip() for b in b.split(",")])
    rating_rdd = reviewRDD.groupByKey().mapValues(lambda ratings: [float(r) for r in ratings]).map(lambda rating: (rating[0], (len(rating[1]), sum(rating[1]))))
    master_rdd = category_rdd.leftOuterJoin(rating_rdd)

    result_rdd = master_rdd.map(lambda key: key[1])\
                .filter(lambda key: key[1] is not None)\
                .flatMap(lambda key_val: [[cat, key_val[1]] for cat in key_val[0]])\
                .reduceByKey(lambda a1,a2: (a1[0]+ a2[0], a1[1] + a2[1]))\
                .mapValues(lambda val: float(val[1]/val[0]))\
                .takeOrdered(n, key=lambda pair: (-pair[1], pair[0]))
    return result_rdd


if is_spark == "spark":
    sc = pyspark.SparkContext("local[*]", "Task2")

    reviews = sc.textFile(review_file)
    reviewRDD = reviews.map(lambda line: json.loads(line)).map(lambda review: (review[B_ID], review[STARS]))

    business = sc.textFile(business_file)
    businessRDD = business.map(lambda line: json.loads(line)).map(lambda review: (review[B_ID], review[CAT]))

    output= average_star_rating(reviewRDD,businessRDD,top_n)
    print(output)
    result['result']=output

else:
    category_dict = dict()  # key = business_id, value = categories
    rating_dict = dict() # key= category, value = [number of ratings, sum of ratings]

    with open(business_file) as File:
        for line in File:
            json_line = json.loads(line)
            category = json_line[CAT]
            b_id = json_line[B_ID]
            if category is not None:
                category_dict[b_id] = category

    with open(review_file) as File:
        for line in File:
            json_line = json.loads(line)
            b_id = json_line[B_ID]
            stars = json_line[STARS]
            if b_id in category_dict:
                categories = category_dict[b_id]
                categories_arr = [b.strip() for b in categories.split(",")]
                #categories_arr = re.split(r'(\s*[,，]\s*)', categories.strip())
                for category in categories_arr:
                    rating_avg = rating_dict.get(category, [0, 0])
                    rating_avg[0] = rating_avg[0] + 1
                    rating_avg[1] = rating_avg[1] + stars
                    rating_dict[category] = rating_avg

    result_array = []
    for category, rating_avg in rating_dict.items():
        result_array.append([category, rating_avg[1]/rating_avg[0]])

    output = heapq.nsmallest(top_n, result_array, key=lambda pair: (-pair[1], pair[0]))
    print(output)
    result['result'] = output

with open(output_file, 'w+') as output_file:
    json.dump(result, output_file)
output_file.close()




