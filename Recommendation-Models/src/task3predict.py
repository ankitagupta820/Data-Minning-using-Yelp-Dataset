import json
from pyspark import SparkContext
import sys
import time
from collections import defaultdict

# constants
BUSINESS_ID = "business_id"
USER_ID = "user_id"
FEATURES = "features"
STARS = "stars"
ITEM_BASED = "item_based"
USER_BASED = "user_based"
B1 = "b1"
B2 = "b2"
SIM = "sim"
sc = SparkContext()
tickTime = time.time()

n = 10

train_file, test_file, model_file, output_file, CF_type = sys.argv[1:]

#local Run
# train_file = "data/HW3/train_review.json"
# test_file = "data/HW3/test_review.json"
# model_file = "data/HW3/task3_output.json"
# output_file = "data/HW3/task3_predictions.json"
# CF_type = "item_based"


def sort(list):
    list.sort(key=lambda l: -l[1])
    return list


#  pred = (predicting user, predicting business), [(business1, rating1), (business2, rating2)]
def item_predict(pred):
    neighbours = []
    user = pred[0][0]
    business = pred[0][1]

    for rating_pair in pred[1]:

        b = rating_pair[0]
        rating = rating_pair[1]

        cand_pair = tuple(sorted((business, b)))
        if model_map.get(cand_pair):
            neighbours.append([rating, model_map[cand_pair]])
    Neighbors_N = sort(neighbours)[:n]

    num = sum([val[0] * val[1] for val in Neighbors_N])
    den = sum([abs(val[1]) for val in Neighbors_N])

    if num != 0 and den != 0:
        star_rating = num / den
    else:
        star_rating = avg_business_rating[business]

    return (user, business, star_rating)

train_data = sc.textFile(train_file).map(lambda line: json.loads(line)).map(lambda row: (row[USER_ID], row[BUSINESS_ID], row[STARS]))
test_data = sc.textFile(test_file).map(lambda line: json.loads(line)).map(lambda row: (row[USER_ID], row[BUSINESS_ID]))
model_data = sc.textFile(model_file).map(lambda line: json.loads(line))

business_avg = train_data.map(lambda l: (l[1], [l[2]])).reduceByKey(lambda a, b: a + b).mapValues(lambda v: (sum(v), len(v))).mapValues(lambda v: v[0] / v[1]).collect()

avg_business_rating = defaultdict(lambda: 0)
for t in business_avg:
    avg_business_rating[t[0]] = t[1]


if CF_type == 'item_based':

    # (user_id,[(business_id, rating)])
    train_rdd = train_data.map(lambda l: (l[0], [(l[1], l[2])])).reduceByKey(lambda a, b: a + b).map(lambda l: (l[0], list(set(l[1]))))


    # (user_id,business_id)
    test_rdd = test_data.map(lambda l: (l[0], l[1]))


    # # (business_id1, business_id2): sim}
    model_array = model_data.map(lambda l: (l[B1], l[B2], l[SIM])).map(lambda l: ((l[0], l[1]), l[2])).collect()
    #
    model_map = dict()
    for List in model_array:
        model_map[tuple(sorted(List[0]))] = List[1]

    # generate predictions
    # (user_id, ([(business, rating)], business_test)) and filter if there is no businesses rated by this user.
    predict_1 = train_rdd.rightOuterJoin(test_rdd).filter(lambda l: l[1][0] != None)

    # ((user_id,business_test), [(business, rating)]))
    predict_2 = predict_1.map(lambda kv: ((kv[0], kv[1][1]), kv[1][0])).reduceByKey(lambda a, b: a + b)

    # calculate rating
    predict_3 = predict_2.map(lambda kv: item_predict(kv)).collect()

    # predict_3 = predict_2.map(lambda kv: (kv[0], [item for list in kv[1] for item in list]))
    with open(output_file, 'w+') as f:
        for item in predict_3:
            f.write(json.dumps({USER_ID: item[0], BUSINESS_ID: item[1], STARS: item[2]}) + '\n')

    tockTime = time.time()
    print('Duration: ', tockTime - tickTime)

else:
    print("User Based CF")



