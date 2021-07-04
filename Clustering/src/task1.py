from pyspark import SparkConf, SparkContext
import os
import sys
import collections
from collections import defaultdict
import itertools
import random
from math import sqrt
import copy
import json
import csv
import time

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

#CONSTANTS
MAHALANOBIS = "m"
EUCLIDEAN = "e"

class Kmeans:

    def __init__(self, n_clusters: int, max_iter: int):
        self.max_iteration = max_iter
        self.n_cluster = n_clusters

    def initialize_centroid(self, s: int):

        """
        Use random sampling to pick points as centroids
        """

        random.seed(s)
        self.centroids = dict()
        self.cluster_points = dict()
        self.centroid_isStable = dict()
        self.centroid_stats = dict()
       
        for index, sample_key in enumerate(random.sample(self.data.keys(), self.n_cluster)):
            self.centroids.setdefault("c" + str(index), self.data.get(sample_key))
            self.centroid_stats.setdefault("c" + str(index), self.data.get(sample_key))
            self.cluster_points.setdefault("c" + str(index), list())
            self.centroid_isStable.setdefault("c" + str(index), False)

    def fit_data(self, data: dict, s=666):

        """
        Performs point assignment to centroids
        """
        self.data = data
        self.check_datasize()  # this func might change the value of self.n_cluster
        self.initialize_centroid(s)
        epochs = 1
        while True:

            for k in list(self.data.keys()):

                # [(centroid, key), distance]
                # => [((c2, point_id), distance_to_c2)]
                distance_to_centroids = dict()
                for c in self.centroids.keys():
                    distance_to_centroids[(c, k)] = findDistance(self.centroids[c], self.data[k])
                assignment = list(sorted(distance_to_centroids.items(), key=lambda x: x[1]))[:1]

                # add the point to corresponding centroid list
                # {cluster_point_id1: [point_id,...], cluster_point_id2: [point_id,...]}
                self.cluster_points[assignment[0][0][0]].append(assignment[0][0][1])

            prev_centroids, curr_centroids = self.update_centroids()

            if not self.is_centroid_changed(prev_centroids, curr_centroids) or epochs >= self.max_iteration:
                break
            epochs += 1
            self.reset_clusters()

        return self.centroids, self.centroid_stats, self.cluster_points

    def check_datasize(self):
        """
        Check the data size. change the number of cluster
        if size of data is less than earlier setup
        """
        if len(self.data.keys()) < self.n_cluster:
            self.n_cluster = len(self.data.keys())

    def reset_clusters(self):

        for key in self.cluster_points.keys():
            self.cluster_points[key] = list()


    def update_centroids(self):
        """
        recalculate the centroids by averaging cluster points
        returns -> previous centroids, new centroids
        """
        prev_centroids = copy.deepcopy(self.centroids)

        for centroid, cluster_point_list in self.cluster_points.items():
            if not self.centroid_isStable.get(centroid):
                points_list = list()
                points_list.append(self.centroids.get(centroid))
                for cluster_point in cluster_point_list:
                    points_list.append(self.data.get(cluster_point))

                # Calculate new centroids by averaging points SUM / N
                self.centroids[centroid] = [sum(d) / len(d) for d in zip(*points_list)]

                # Calculate SUMSQ / N
                self.centroid_stats[centroid] = [sum([val ** 2 for val in i]) / len(i) for i in
                                                     zip(*points_list)]

        return prev_centroids, self.centroids

    def is_centroid_changed(self, c1: dict, c2: dict):

        for id in c1.keys():
            valA = set(map(lambda v: round(v, 0), c1.get(id)))
            valB = set(map(lambda v: round(v, 0), c2.get(id)))

            if len(valA.difference(valB)) == 0:
                self.centroid_isStable[id] = True
            else:
                self.centroid_isStable[id] = False
                return True

        return False

class Cluster:

    def __init__(self):
        self.centroids = None
        self.cluster_points = None
        self.Type = None

    def init(self, centroids: dict, stats: dict, points: dict):
        self.data_dimentions = len(list(centroids.values())[0])
        self.cluster_points = points
        self.centroids = centroids
        self.SUM_N = centroids
        self.SUMSQ_N = stats
        self.total_point = 0
        self.STD = dict()
        self.calc_STD()

    def getType(self):
        return self.Type

    def getClusterPoints(self):
        return self.cluster_points

    def getClusterPointsByKey(self, key):
        return list(self.cluster_points.get(key))

    def getCentroids(self):
        return self.centroids

    def getCentroidByKey(self, key: str):
        return list(self.centroids.get(key))

    def getSTD(self):
        return self.STD

    def getSTDByKey(self, key: str):
        return list(self.STD.get(key))

    def getDimension(self):
        return self.data_dimentions

    def getNumClusters(self):
        return len(self.centroids.keys())

    def getNumPoints(self):
        self.total_point = 0
        for key, value in self.cluster_points.items():
            if type(value) == list:
                self.total_point += len(value)
        return self.total_point

    def getSUMSQ_NByKey(self, key: str):
        return list(self.SUMSQ_N.get(key))

    def calc_STD(self):
        self.STD = dict()
        for k in self.SUM_N.keys():
            self.STD[k] = [sqrt(SQ_N - SUM_N ** 2) for (SQ_N, SUM_N) in zip(self.SUMSQ_N.get(k), self.SUM_N.get(k))]

    def updateCentroids(self, cluster_map, data_point_map):

        if len(cluster_map.keys()) > 0:

            old_centroids = copy.deepcopy(self.centroids)
            old_cluster_points = copy.deepcopy(self.cluster_points)
            old_sumsq_n = copy.deepcopy(self.SUMSQ_N)

            for centroid, points in cluster_map.items():

                new_point_list = list()
                for point in points:
                    new_point_list.append(data_point_map.get(point))

                n_old_points = len(old_cluster_points.get(centroid))
                total_count = n_old_points + len(new_point_list)

                # Update value for SUM/N
                old_location1 = old_centroids.get(centroid)
                old_sum = list(map(lambda val: val * n_old_points, old_location1))
                new_sum = [sum(d) for d in zip(*new_point_list)]
                self.centroids[centroid] = self.SUM_N[centroid] = computeAVG(old_sum, new_sum, denom=total_count)

                # update value for SUMSQ / N
                old_location2 = old_sumsq_n.get(centroid)
                old_sumsq = list(map(lambda val: val * n_old_points, old_location2))
                new_sumsq = [sum([val ** 2 for val in i]) for i in zip(*new_point_list)]
                self.SUMSQ_N[centroid] = computeAVG(new_sumsq, old_sumsq, denom=total_count)

            self.calc_STD()
            self.updateClusterPoints(cluster_map)

    def updateClusterPoints(self, new_cluster_points):

        if len(new_cluster_points.keys()) > 0:
            total_list = collections.defaultdict(list)
            for key, val in itertools.chain(self.cluster_points.items(), new_cluster_points.items()):
                total_list[key] += val
            self.cluster_points = total_list

def computeAVG(l1, l2, denom=None):
    L = list()
    L.append(l1)
    L.append(l2)

    if denom is None:
        return [sum(d) / len(d) for d in zip(*L)]
    else:
        return [sum(d) / denom for d in zip(*L)]


class DS(Cluster):

    def __init__(self):
        Cluster.__init__(self)
        self.Type = "DS"

    def mergeToOneCluster(self, ds_key: str,
                          cs_sumsq_n: list,
                          cs_centroid: list,
                          cs_points: list):

        n_ds_points = len(self.getClusterPointsByKey(ds_key))
        n_cs_points = len(cs_points)
        total_points = n_cs_points + n_ds_points

        ds_centroid = self.getCentroidByKey(ds_key)

        old_sum = list(map(lambda val: val * n_ds_points, ds_centroid))
        new_sum = list(map(lambda val: val * n_cs_points, cs_centroid))
        sum = computeAVG(old_sum, new_sum, total_points)

        old_sumsq = list(map(lambda val: val * n_ds_points, self.getSUMSQ_NByKey(ds_key)))
        new_sumsq = list(map(lambda val: val * n_cs_points, cs_sumsq_n))
        sumsq = computeAVG(old_sumsq, new_sumsq, total_points)


        self.centroids.update({ds_key: sum})
        self.cluster_points[ds_key].extend(cs_points)
        self.SUMSQ_N.update({ds_key: sumsq})
        self.getNumPoints()
        self.calc_STD()


class CS(Cluster):

    def __init__(self):
        Cluster.__init__(self)
        self.Type = "CS"
        self.R2C_itr = 0
        self.merge_itr = 0

    def removeCluster(self, key):

        # pop cluster points
        self.cluster_points.pop(key)

        # pop centroid and SUM_N
        self.centroids.pop(key)

        # pop standard dev for cluster
        self.STD.pop(key)

        # pop sum of squares avg.
        self.SUMSQ_N.pop(key)

        # recount total points
        self.getNumPoints()

    def update_change(self, centroids: dict, stats: dict, points: dict):

        if len(centroids.keys()) != 0:
            for key in list(centroids.keys()):
                self.centroids.update({"R2C" + str(self.R2C_itr): centroids.get(key)})
                self.SUMSQ_N.update({"R2C" + str(self.R2C_itr): stats.get(key)})
                self.cluster_points.update({"R2C" + str(self.R2C_itr): points.get(key)})
                self.calc_STD()
                self.R2C_itr += 1


    def merge_two_Clusters(self, C1: str, C2: str):

        new_centroid = computeAVG(list(self.centroids[C1]), list(self.centroids[C2]))
        new_sumsq_n = computeAVG(list(self.SUMSQ_N[C1]), list(self.SUMSQ_N[C2]))

        cluster_points = list(self.cluster_points[C1])
        cluster_points.extend(list(self.cluster_points[C2]))

        m_itr_key = "m" + str(self.merge_itr)

        # Update centroids
        self.centroids.pop(C1)
        self.centroids.pop(C2)
        self.centroids.update({m_itr_key: new_centroid})

        # Update SUMSQ_N
        self.SUMSQ_N.pop(C1)
        self.SUMSQ_N.pop(C2)
        self.SUMSQ_N.update({m_itr_key: new_sumsq_n})

        # Update cluster points
        self.cluster_points.pop(C1)
        self.cluster_points.pop(C2)
        self.cluster_points.update({m_itr_key: cluster_points})

        # Recalculate STD
        self.calc_STD()
        self.merge_itr += 1

    def getClusterResultSortedInfo(self):
        result = collections.defaultdict(list)
        for key in self.cluster_points.keys():
            result[key] = sorted(self.cluster_points[key])

        return result


class RS:

    def __init__(self):
        self.remaining_set = dict()
        self.Type = "RS"

    @classmethod
    def getType(cls):
        return "RS"

    def addPoints(self, data: dict):
        self.remaining_set.update(data)

    def countPoints(self):
        return len(self.remaining_set.keys())

    def getRemaining(self):
        return self.remaining_set

    def setPoints(self, points: dict):
        self.remaining_set = points


class IntermediateSteps:

    def __init__(self):
        self.intermediate_steps = dict()
        self.intermediate_steps["header"] = (
            "round_id", "nof_cluster_discard", "nof_point_discard",
            "nof_cluster_compression", "nof_point_compression",
            "nof_point_retained"
        )

    def add_intermediate_step(self, round_no, ds, cs, rs):
        self.intermediate_steps[round_no] = ( round_no, ds.getNumClusters(), ds.getNumPoints(), cs.getNumClusters(), cs.getNumPoints(), rs.countPoints())
        print("Round "+str(round_no)+" -> DS: Clusters: "+str(ds.getNumClusters())+" Points: "+str(ds.getNumPoints())+" | CS: Clusters: "+str(cs.getNumClusters())+" Points: "+str(cs.getNumPoints())+" | RS: Points: "+str(rs.countPoints())+"")

    def write(self, path: str):
        writeToFile(self.intermediate_steps, path, type="csv")


def writeToFile(results, path, type="json"):

    if type == "json":
        with open(path, "w+") as f:
            f.writelines(json.dumps(results))
            f.close()

    elif type == "csv":
        with open(path, "w+", newline="") as f:
            writer = csv.writer(f)
            for key, value in results.items():
                writer.writerow(value)


def segregate_single_point_clusters(centroids: dict, statistics: dict, points: dict):

   # remove clusters and points with 1 point
    remaining_points = dict()
    temp_cluster_result = copy.deepcopy(points)
    for centroid, cluster_points in temp_cluster_result.items():
        if len(cluster_points) <= 1:
            if len(cluster_points) != 0:
                remaining_points.update({cluster_points[0]: centroids.get(centroid)})
            points.pop(centroid)
            centroids.pop(centroid)
            statistics.pop(centroid)

    return centroids, statistics, points, remaining_points


def findDistance(pointX, pointY, STD = None, d_type= EUCLIDEAN):

    if d_type == EUCLIDEAN:
        return float(sqrt(sum([(x - y) ** 2 for (x, y) in zip(pointX, pointY)])))
    elif d_type == MAHALANOBIS:
        return float(sqrt(sum([((x - y) / std) ** 2 for (x, y, std) in zip(pointX, pointY, STD)])))


def assign2NearestCluster(data_point, alpha, DS=None, CS=None, c_type=""):

    if DS is not None and c_type == DS.getType():
        ds_dimensions = DS.getDimension()
        min_distance = float('inf')
        closest_key = None
        for key, point in DS.getCentroids().items():
            curr_distance = findDistance(data_point[1], point, DS.getSTD().get(key), d_type = MAHALANOBIS)
            if curr_distance < alpha * sqrt(ds_dimensions) and curr_distance < min_distance:
                min_distance = curr_distance
                closest_key = (key, data_point[0])

        if closest_key is not None:
            # assigned to closest DS
            yield tuple((closest_key, data_point[1], False))
        else:
            #Outlier point
            yield tuple((("-1", data_point[0]), data_point[1], True))

    elif CS is not None and c_type == CS.getType():
        cs_dimensions = CS.getDimension()
        min_distance = float('inf')
        closest_key = None
        for key, point in CS.getCentroids().items():
            curr_distance = findDistance(data_point[1], point, CS.getSTD().get(key), d_type=MAHALANOBIS)
            if curr_distance < alpha * sqrt(cs_dimensions) and curr_distance < min_distance:
                min_distance = curr_distance
                closest_key = (key, data_point[0])

        if closest_key is not None:
            # assigned to closest CS
            yield tuple((closest_key, data_point[1], False))
        else:
            # Outlier point
            yield tuple((("-1", data_point[0]), data_point[1], True))


def merge_CS_Clusters(alpha, cs: CS):

    cs_D = cs.getDimension()
    old_cs = copy.deepcopy(cs)

    centroid_keys = set(list(old_cs.getCentroids().keys()))

    for pair in itertools.combinations(list(old_cs.getCentroids().keys()), 2):

        if pair[0] in centroid_keys and pair[1] in centroid_keys:

            distance = findDistance(pointX=old_cs.getCentroidByKey(pair[0]), pointY=old_cs.getCentroidByKey(pair[1]), STD=old_cs.getSTDByKey(pair[0]), d_type=MAHALANOBIS)

            if distance < alpha * sqrt(cs_D):
                # Merge 2 CS
                cs.merge_two_Clusters(pair[0], pair[1])
                centroid_keys.discard(pair[0])
                centroid_keys.discard(pair[1])


def merge_CS_DS(alpha_value, ds: DS, cs: CS):

    ds_D = ds.getDimension()

    old_ds = copy.deepcopy(ds)
    old_cs = copy.deepcopy(cs)

    for c_cs in old_cs.getCentroids().keys():

        for c_ds in old_ds.getCentroids().keys():

            cs_ds_distance = findDistance(
                pointX=old_cs.getCentroidByKey(c_cs),
                pointY=old_ds.getCentroidByKey(c_ds),
                STD=old_ds.getSTDByKey(c_ds),
                d_type= MAHALANOBIS
            )

            if cs_ds_distance < alpha_value * sqrt(ds_D):

                ds.mergeToOneCluster(ds_key=c_ds,
                                     cs_sumsq_n=old_cs.getSUMSQ_NByKey(c_cs),
                                     cs_centroid=old_cs.getCentroidByKey(c_cs),
                                     cs_points=old_cs.getClusterPointsByKey(c_cs))
                cs.removeCluster(c_cs)
                break


def WriteClusterOutput(ds: DS, cs: CS, rs: RS, path):

    result = defaultdict()

    for key in list(ds.getClusterPoints().keys()):
        [result.setdefault(str(id), int(key[1:])) for id in ds.getClusterPointsByKey(key)]

    for key in list(cs.getClusterPoints().keys()):
        [result.setdefault(str(id), -1) for id in cs.getClusterPointsByKey(key)]

    for key in list(rs.getRemaining().keys()):
        result.setdefault(str(key), -1)

    writeToFile(result, path, type="json")



tick = time.time()

#input paths and variables
input_dir = "data/HW5/test1"
n_clusters = int("10")
outfile_1 = "data/HW5/output/cluster1.json"
outfile_2 = "data/HW5/intermediate/intermediate1.csv"

# input_dir = sys.argv[1]
# n_clusters = int(sys.argv[2])
# outfile_1 = sys.argv[3]
# outfile_2 = sys.argv[4]

conf = SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

alpha = 3
discard_set = DS()
compressed_set = CS()
retained_set = RS()
intermediate_steps = IntermediateSteps()

for i, file_name in enumerate(sorted(os.listdir(input_dir))):

    file_path = ''.join(input_dir + "/" + file_name)
    Data_RDD = sc.textFile(file_path).map(lambda line: line.split(",")).map(lambda x: (int(x[0]), list(map(eval, x[1:]))))

    if i == 0:
        total_points = Data_RDD.count()
        first_n = 10000 if total_points > 10000 else int(total_points * 0.1)

        # Run KMeans on subset of first data file.
        kmean_data = Data_RDD.filter(lambda x: x[0] < first_n).collectAsMap()
        ds_centeroid, ds_stats, ds_points = Kmeans(n_clusters=n_clusters, max_iter=5).fit_data(kmean_data)

        # Run K-Means to generate the DS clusters
        discard_set.init(ds_centeroid, ds_stats, ds_points)

        # run kMeans on rest of points to get CS
        remaining_data = Data_RDD.filter(lambda x: x[0] >= first_n).collectAsMap()
        centeroid, stats, points = Kmeans(n_clusters=n_clusters * 3, max_iter=3).fit_data(remaining_data)

        # seperate points into RS.
        cs_centeroid, cs_stats, cs_points, remaining = segregate_single_point_clusters(centeroid, stats, points)
        compressed_set.init(cs_centeroid, cs_stats, cs_points)
        retained_set.addPoints(remaining)

    else:
        # Compare new points to the clusters in DS and if the distance < 𝛼√𝑑,
        # assign them to the nearest DS cluster.

        # For assigned points (('-1', point_index), point_location, True)
        # For Outliers points (('c1', point_index), point_location, False)
        All_Assignment_RDD = Data_RDD.flatMap(lambda data_point: assign2NearestCluster(data_point, alpha, DS=discard_set, c_type=discard_set.getType()))

        # Get all assigned points (('c1', point_index), point_location)
        DS_Assignment_RDD = All_Assignment_RDD.filter(lambda assignment: assignment[2] is False).map(lambda x: (x[0], x[1]))

        # Group by DS cluster name {"c1": [point_index1, point_index2....]}
        ds_cluster_map = DS_Assignment_RDD.map(lambda x: x[0]).groupByKey().mapValues(list).collectAsMap()

        # Point map {point_index: point_location}
        ds_data_points = DS_Assignment_RDD.map(lambda x: (x[0][1], list(x[1]))).collectAsMap()

        # Update DS centroids with new data points
        discard_set.updateCentroids(ds_cluster_map, ds_data_points)

        # Assign the outlier points of from DS assignment to nearest CS based on Mahalanobis Distance
        # => output (('centroid', point_index), location, False)
        All_Assignment_CS = All_Assignment_RDD.filter(lambda assignment: assignment[2] is True).map(lambda x: (x[0][1], x[1]))\
            .flatMap(lambda data_point: assign2NearestCluster(data_point, alpha, CS=compressed_set, c_type=compressed_set.getType()))

        #(('centroid', point_index), location)
        CS_Assignment_RDD = All_Assignment_CS.filter(lambda x: x[2] is False).map(lambda x: (x[0], x[1]))

        # Group by CS cluster name {"c1": [point_index1, point_index2....]}
        cs_cluster_map = CS_Assignment_RDD.map(lambda x: x[0]).groupByKey().mapValues(list).collectAsMap()

        # Point map {point_index: point_location}
        cs_data_points = CS_Assignment_RDD.map(lambda x: (x[0][1], list(x[1]))).collectAsMap()

        # Update CS centroids with new data points
        compressed_set.updateCentroids(cs_cluster_map, cs_data_points)

        # Assign the outlier points from CS assignment to RS
        # {point_index: location}
        remaining_point = All_Assignment_CS.filter(lambda assignment: assignment[2] is True).map(lambda x: (x[0][1], x[1])).collectAsMap()
        retained_set.addPoints(remaining_point)

        # Run K-Means on RS points to get CS (>1 point) and RS (<=1)
        centeroids, stats, points = Kmeans(n_clusters=n_clusters * 3, max_iter=5).fit_data(retained_set.getRemaining())

        cs_centeroids, cs_stats, cs_points, other = segregate_single_point_clusters(centeroids, stats, points)
        compressed_set.update_change(cs_centeroids, cs_stats, cs_points)
        retained_set.setPoints(other)

        # Merge CS clusters if their mahalanobis distance < 𝛼√𝑑
        merge_CS_Clusters(alpha, cs=compressed_set)


    # merge CS clusters to  DS if their mahalanobis distance < 𝛼√𝑑
    if len(os.listdir(input_dir)) == i + 1:
        merge_CS_DS(alpha, ds=discard_set, cs=compressed_set)

    intermediate_steps.add_intermediate_step(i + 1, discard_set, compressed_set, retained_set)

# Write to Output Files
intermediate_steps.write(outfile_2)
WriteClusterOutput(ds=discard_set, cs=compressed_set, rs=retained_set, path=outfile_1)
print("Duration: %d s." % (time.time() - tick))