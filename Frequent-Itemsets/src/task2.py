import pyspark
import sys
import itertools
import time

#Initialize Spark Context
sc = pyspark.SparkContext("local[*]", "Task1")
sc.setLogLevel("ERROR")
startTime = time.time()

# Program inputs
# filter = 100
# threshold = 45
# input_file = "data/HW2/user-business.csv"
# output_file = "data/HW2/task2_output.csv"
# task2_numbers_output = "data/HW2/task2_numbers.csv"
# task2_baskets_output = "data/HW2/task2_baskets.csv"


filter = int(sys.argv[1])
threshold = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]
task2_numbers_output = "task2_numbers.csv"
task2_baskets_output = "task2_baskets.csv"

#Defined constants
USER_ID = "user_id"
BUSINESS_ID = "business_id"

def count_frequency(itemSet, baskets, scaled_threshold):
    count =0
    for b in baskets:
        if(set(itemSet).issubset(b)):
            count+=1
            if(count>=scaled_threshold):
                return count
    return count


def APriori(baskets_chunk, scaledthreshold):

    Baskets = list(baskets_chunk)
    itemSet = list(set().union(*Baskets))

    frequent_itemsets = []
    List1 = []
    List2 = []

    #-------------------------------------------frequent itemsets of size 1 --------------------------------------------
    setSize = 1
    for item in itemSet:
        count = 0
        for basket in Baskets:
            if (item in basket):
                count=count+1
                if count >= scaledthreshold:
                    List2.append(item)
                    break
        List1.append(item)

    List1.sort()
    List2.sort()
    l1_len = len(List2)

    S1=[(i,) for i in List2]
    frequent_itemsets.extend(S1)

    #---------------------------------frequent itemsets of size 2-------------------------------------------------------
    setSize+=1
    List1.clear()

    for combination in itertools.combinations(List2,2):
        pair=list(combination)
        pair.sort()
        List1.append(pair)

    List1.sort()
    List2.clear()

    for pair in List1:
        frequency = count_frequency(pair, Baskets, scaledthreshold)
        if (frequency >= scaledthreshold):
            List2.append(pair)

    List2.sort()
    frequent_itemsets.extend(List2)

    #---------------------------------frequent itemsets of size 2+ --------------------------------------------------
    setSize += 1
    while (setSize != l1_len):

        List1.clear()
        List1 = largerFrequentCandidates(List2, setSize)
        if(len(List1) == 0):
            break
        List2.clear()
        List1.sort()


        for itemSet in List1:
            frequency = count_frequency(itemSet, Baskets, scaledthreshold)
            if(frequency >= scaledthreshold):
                List2.append(itemSet)

        List2.sort()
        frequent_itemsets.extend(List2)
        setSize += 1


    return frequent_itemsets


def largerFrequentCandidates(Y, setSize):

    candidates =[]

    for i in range(len(Y) - 1):
        for j in range(i + 1, len(Y)):
            if (Y[i][0:setSize - 2] == Y[j][0:setSize - 2]):
                c = list(set(Y[i]) | set(Y[j]))
                c.sort()
                if (c not in candidates):
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
input_rdd = input_text.map(lambda l: l.split(',')).map(lambda l: (l[0],l[1])).distinct().map(lambda l: (l[0],[l[1]]))
model_RDD = input_rdd.reduceByKey(lambda a, b: a+b).persist().filter(lambda l: l[0] != USER_ID).map(lambda l: set(l[1])).filter(lambda x: len(x) > filter)

i_in_baskets= model_RDD.collect()
itemList = list(set().union(*i_in_baskets))


#get Number of partitions of RDD
num_partitions=model_RDD.getNumPartitions()
scaledThreshold = threshold/num_partitions
#print("Num partitions:", num_partitions)


# SON Phase 1 - Identify candidate frequent Items using Apriori
candidateItemsets = model_RDD.mapPartitions(lambda chunk_baskets: APriori(chunk_baskets, scaledThreshold)).map(lambda item: (tuple(item),1))
candidates = candidateItemsets.distinct().sortByKey().map(lambda itemset: itemset[0]).collect()

# SON Phase 2 - Count frequent candidates in entire dataset
trueFrequentItemsets = model_RDD.mapPartitions(lambda chunk_baskets: findTrueFrequents(chunk_baskets,candidates))
trueFrequents = trueFrequentItemsets.reduceByKey(lambda x,y: x+y).filter(lambda itemset: itemset[1] >= threshold).sortByKey().keys()


# write the output file
with open(output_file, 'w') as ofile:
	ofile.write('Candidates:' + '\n')
	outputFile(candidates, ofile)

	ofile.write('Frequent Itemsets:' + '\n')
	outputFile(trueFrequents.collect(), ofile)


endTime = time.time()
print("Duration: ", endTime-startTime)

#Save total number of frequent baskets for task 3
count = trueFrequents.count()
with open(task2_numbers_output, "w") as task2_num_file:
    task2_num_file.write(str(count))

#Save baskets for task 3
with open(task2_baskets_output, "w") as task2_baskets_file:
    for basket in trueFrequents.collect():
        task2_baskets_file.write(",".join(basket)+"\n")

