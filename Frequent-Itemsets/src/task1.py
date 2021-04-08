import pyspark
import sys
import itertools
import time

#Initialize Spark Context
sc = pyspark.SparkContext("local[*]", "Task1")
sc.setLogLevel("ERROR")

startTime = time.time()

#Program inputs
# model_case = 2
# threshold = 9
# input_file = "data/HW2/small2.csv"
# output_file = "data/output.json"

model_case = int(sys.argv[1])
threshold = int(sys.argv[2])
input_file =sys.argv[3]
output_file = sys.argv[4]

#Defined constants
USER_ID = "user_id"
BUSINESS_ID = "business_id"

def count_frequency(itemSet, baskets):
    count = 0
    for b in baskets:
        if(set(itemSet).issubset(b)):
            count += 1
    return count


def APriori(baskets_chunk, scaledthreshold):

    Baskets = list(baskets_chunk)
    itemSet = list(set().union(*Baskets))

    frequent_itemsets = []
    X = []
    Y = []

    # Frequent itemsets of size 1
    setSize = 1
    for item in itemSet:
        count=0
        for basket in Baskets:
            if (item in basket):
                count = count+1
                if count >= scaledthreshold:
                    Y.append(item)
                    break
        X.append(item)

    X.sort()
    Y.sort()
    l1_len = len(Y)

    S1 = [(i,) for i in Y]
    frequent_itemsets.extend(S1)

    # Frequent itemsets of size 2
    setSize += 1
    X.clear()

    for combination in itertools.combinations(Y,2):
        pair = list(combination)
        pair.sort()
        X.append(pair)

    X.sort()
    Y.clear()

    for pair in X:
        frequency = count_frequency(pair,Baskets)
        if (frequency >= scaledthreshold):
            Y.append(pair)

    Y.sort()
    frequent_itemsets.extend(Y)

    #Frequent itemsets of size 2+
    setSize += 1
    while (setSize != l1_len):

        X.clear()
        X = largerFrequentCandidates(Y, setSize)
        if(len(X) == 0):
            break
        X.sort()
        Y.clear()

        for itemSet in X:
            frequency = count_frequency(itemSet,Baskets)
            if(frequency >= scaledthreshold):
                Y.append(itemSet)

        Y.sort()
        frequent_itemsets.extend(Y)
        setSize += 1

    return frequent_itemsets


def largerFrequentCandidates(Y, setSize):

    candidates = []

    for i in range(len(Y) - 1):
        for j in range(i + 1, len(Y)):
            if (Y[i][0:setSize - 2] == Y[j][0:setSize - 2]):
                c = list(set(Y[i]) | set(Y[j]))
                c.sort()
                if c not in candidates:
                    candidates.append(c)
            else:
                break

    return candidates

def findTrueFrequents(chunk_basket, candidates):

    baskets = list(chunk_basket)
    trueFrequents = []

    for candidate in candidates:
        counter = 0
        for basket in baskets:
            if ((set(candidate)).issubset(basket)):
                counter += 1
        trueFrequents.append([candidate, counter])

    return trueFrequents

def outputFile(frequentItemsets, outputFile):

    itemSet_length = 1
    while itemSet_length != len(itemList):
        k_length_sets = ""
        for r in frequentItemsets:
            if (len(r) == itemSet_length):
                k_length_sets = k_length_sets + str(r)

        k_length_sets = k_length_sets.replace(")(", "),(").replace(",)", ")")
        if (k_length_sets == ""):
            break
        else:
            if (itemSet_length != 1):
                outputFile.write("\n\n")
            outputFile.write(k_length_sets)
        itemSet_length = itemSet_length + 1
    outputFile.write("\n\n")

input_text = sc.textFile(input_file)

# define market-basket model
if(model_case == 1):
    input_rdd = input_text.map(lambda l: l.split(','))\
        .map(lambda l: (l[0],l[1])).distinct()\
        .map(lambda l: (l[0],[l[1]]))

    model_RDD = input_rdd.reduceByKey(lambda a,b: a+b)\
        .filter(lambda l: l[0] != USER_ID)\
        .map(lambda l: l[1])
else:
    input_rdd = input_text.map(lambda l: l.split(','))\
        .map(lambda l: (l[0],l[1])).distinct()\
        .map(lambda l: (l[1], l[0]))\
        .map(lambda l: (l[0], [l[1]]))

    model_RDD = input_rdd.reduceByKey(lambda a,b: a+b).persist()\
        .filter(lambda l: l[0] != BUSINESS_ID)\
        .map(lambda l: l[1])

#print("Baskets: ",model_RDD.collect())
i_in_baskets = model_RDD.collect()
itemList = list(set().union(*i_in_baskets))


#get Number of partitions of RDD
num_partitions = model_RDD.getNumPartitions()
scaledThreshold = threshold/num_partitions



# SON Phase 1 - Identify candidate frequent Items using Apriori
candidateItemsets = model_RDD.mapPartitions(lambda chunk_baskets: APriori(chunk_baskets, scaledThreshold)).map(lambda item: (tuple(item), 1))
candidates = candidateItemsets.distinct().sortByKey().map(lambda itemset: itemset[0]).collect()

# SON Phase 2 - Count frequent candidates in entire dataset
trueFrequentItemsets = model_RDD.mapPartitions(lambda chunk_baskets: findTrueFrequents(chunk_baskets,candidates))
trueFrequents = trueFrequentItemsets.reduceByKey(lambda x, y: x+y).filter(lambda itemset: itemset[1] >= threshold).sortByKey().keys().collect()

#print(trueFrequents)

# write the output file
with open(output_file, 'w') as ofile:
	ofile.write('Candidates:' + '\n')
	outputFile(candidates, ofile)

	ofile.write('Frequent Itemsets:' + '\n')
	outputFile(trueFrequents, ofile)
#print(TrueFrequents)

endTime = time.time()

print("Duration: ", endTime-startTime)