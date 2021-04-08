import pyspark
import json
import sys
from datetime import datetime
from operator import add

def exclude(word):
    if word not in stop_words:
        return ''.join(char for char in word if char not in excluded)


input_file="data/review.json"
output_file="data/task1_ans"
stop_word_file="data/stopwords"
top_m = 5
year = 2005
top_n = 10


# input_file = sys.argv[1]
# output_file = sys.argv[2]
# stop_word_file = sys.argv[3]
# year = int(sys.argv[4])
# top_m = int(sys.argv[5])
# top_n= int(sys.argv[6])
# print("reached here")

excluded = set("!?(),.[]:;")
stopwords = open(stop_word_file)
stop_words = set(w.strip() for w in stopwords)
output = dict()


sc = pyspark.SparkContext("local[*]", "Task1")
sc.setLogLevel("ERROR")
textRDD = sc.textFile(input_file)
reviewsRDD = textRDD.map(lambda l: json.loads(l)).persist(pyspark.StorageLevel.MEMORY_AND_DISK_2)

#Task 1.A 1151625
review_idRDD = reviewsRDD.map(lambda review: review["review_id"])
totalReviews = review_idRDD.count()
print("Total reviews:", totalReviews)
output["A"]=totalReviews

#Task 1.B
rev_date_rdd = reviewsRDD.map(lambda review: (review['review_id'], review['date']))
review_count = rev_date_rdd.filter(lambda review: datetime.strptime(review[1], '%Y-%m-%d %H:%M:%S').year == year).count()
print("Total reviews in year", year, "is: ", review_count)
output["B"] = review_count

#Task 1.C
user_id_rdd = reviewsRDD.map(lambda review: review['user_id'])
user_count = user_id_rdd.distinct().count()
print("Unique User count:", user_count)
output["C"] = user_count

#Task 1.D
users_id_rdd = reviewsRDD.map(lambda review: (review['user_id'],1))
top_m_users= users_id_rdd.reduceByKey(add).sortByKey(False).takeOrdered(top_m, key= lambda review: (-review[1], review[0]))
top_users=list(map(lambda user: [user[0], user[1]], top_m_users))
print(top_users)
output["D"] = top_users


#Task 1.E
review_rdd = reviewsRDD.map(lambda review: review['text']).flatMap(lambda t: t.lower().split(' '))
words_rdd = review_rdd.map(lambda w: (exclude(w), 1)).filter(lambda word:  (word[0] is not "") and (word[0] is not None))
top_rdd = words_rdd.reduceByKey(add).takeOrdered(top_n, key=lambda word: -word[1])
top_words=list(map(lambda kv: kv[0], top_rdd))
print(top_words)
output["E"] = top_words

print(output)

with open(output_file,'w') as f:

    json.dump(output,f)
f.close()
