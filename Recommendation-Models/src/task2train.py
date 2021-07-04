import sys
from pyspark import SparkContext
import time
import json
import re
import math
from collections import defaultdict


tickTime = time.time()
sc = SparkContext()
tickTime = time.time()

#local files
# input_file = "data/HW3/train_review.json"
# output_file = "data/HW3/task2_output.json"
# stopwords_file = "data/HW3/stopwords"

#sys args
input_file, output_file, stopwords_file = sys.argv[1:]

#constants
BUSINESS_ID = "business_id"
USER_ID = "user_id"
REVIEW="text"
FEATURES="features"

def termFrequency(wordslist):

    freq_count = defaultdict(int)
    termFrequency = list()

    # calculate freq of each word
    for w in wordslist:
        freq_count[w] += 1

    #find TF of each word
    max_freq = max(freq_count.values())
    for w, count in freq_count.items():
        termFrequency.append([w, count/max_freq])

    return termFrequency



def wordList(review_text, stopwords):

    rev_text = re.sub(r'[^\w\s]', ' ', review_text)
    rev_text = ''.join(ch for ch in rev_text if not ch.isdigit())
    list = rev_text.split()
    cleaned_list = [w for w in list if w not in stopwords]
    return cleaned_list

def sortList(list):
	list.sort(key=lambda kv: -kv[1])
	return list

tickTime = time.time()

#read files
stopwords=list()
with open(stopwords_file,"r") as stopwords_f:
    stopwords = [row.strip() for row in stopwords_f]


review_data = sc.textFile(input_file).map(lambda l: json.loads(l))
RDD = review_data.map(lambda l: (l[USER_ID], l[BUSINESS_ID], l[REVIEW]))
businesses = RDD.map(lambda l: l[1]).distinct().collect()
n_B = len(businesses)

# generate rdd with [business -> word list for all reviews]
bussiness_rwords = RDD.map(lambda l: (l[1], l[2].lower())).map(lambda l: (l[0], wordList(l[1], stopwords))).groupByKey().map(lambda kv: (kv[0], [w for list in kv[1] for w in list]))

# Term Frequency (TF)
# (business, [(word,count)]) // (business, (word,count)) //(word, (business, count))
tf_rdd = bussiness_rwords.mapValues(lambda v: termFrequency(v)).flatMap(lambda kv: [(kv[0], list) for list in kv[1]]).map(lambda x:(x[1][0], (x[0], x[1][1])))

# Inverse document frequency (IDF)
# (business, [(word,count)]) // [(word,business)] // (word, [business]) // (word, idf)
idf_rdd = bussiness_rwords.flatMap(lambda kv: [(w, kv[0]) for w in kv[1]]).groupByKey().mapValues(lambda v: math.log2(n_B/len(set(v))))

# (word, ((business, tf), idf)) every word repeats for every business. // (business, (word, tf.idf)) // (business, [(word, tf.idf)]) //  (business, [(word,tf.idf)]) top 200
tf_idf_rdd = tf_rdd.join(idf_rdd).map(lambda kkv: (kkv[1][0][0], (kkv[0],  kkv[1][1] * kkv[1][0][1]))).groupByKey().map(lambda kv: (kv[0], [list for list in sortList(list(kv[1]))[:200]]))


# indexing of top all top rated words.
words_map = dict()
wordsList = tf_idf_rdd.flatMap(lambda kv: [kv[0] for kv in kv[1]]).distinct().collect()

for w in range(0,len(wordsList)):
	words_map[wordsList[w]] = w

# business profile (business, [(word_id, tf.idf)])
profile_business = tf_idf_rdd.map(lambda kv:(kv[0], [(words_map[list[0]], list[1]) for list in kv[1]])).collect()

# convert profile_business to Map
business_profile_map = dict()
for item in profile_business:
		business_profile_map[item[0]] = item[1]

# create user profiles

#(user,business) // (user, [business]) // (user, [unique_business]) // (user, [ for each bsuiness [(word_id,tf.idf)]]) // (user, (word_id, tf.idf))
user_profile = RDD.map(lambda i: (i[0],i[1])).groupByKey().map(lambda kv: (kv[0], [b for b in set(kv[1])])).map(lambda kv: (kv[0], [business_profile_map[b] for b in kv[1]])).map(lambda kv: (kv[0], list(set([w for list in kv[1] for w in list])))).map(lambda kv: (kv[0], [list[0] for list in sortList(kv[1])[:500]])).collect()

with open(output_file, 'w+') as f:
		for row in profile_business:
			f.write(json.dumps({BUSINESS_ID: row[0], FEATURES : [list[0] for list in row[1]]}) + '\n')
		for row in user_profile:
			f.write(json.dumps({USER_ID: row[0], FEATURES: row[1]}) + '\n')

tockTime = time.time()
print("Duration: ", tockTime-tickTime)

