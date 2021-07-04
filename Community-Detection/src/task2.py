import sys
import time
import operator
from pyspark import SparkConf
from pyspark import SparkContext
from collections import defaultdict
from itertools import combinations
from operator import add

def getEdges(pair):

    edges = []
    if len(set(UB_map[pair[0]]).intersection(set(UB_map[pair[1]]))) >= threshold:
        edges.append(tuple(pair))

    return edges

def calc_betweenness(vertices, graph):

    result = {}

    for rootNode in vertices:

        visited, shortestPaths, parents = BFS(rootNode, graph)
        edge_credits = find_edge_credits(visited, shortestPaths, parents)
        for e, c in edge_credits.items():
            if e not in result.keys():
                result[e] = 0
            result[e] += c/2
    return sorted(result.items(), key = operator.itemgetter(1), reverse=True)


def BFS(root_node, graph):

    q = [root_node]
    visited = [root_node]
    level = {}
    parents = {}
    shortestPath = {}

    # initialize values for all vertices
    for v in graph.keys():
        level[v] = -1
        parents[v] = []
        shortestPath[v] = 0

    # initialise values for root element
    level[root_node] = 0
    shortestPath[root_node] = 1

    while q:

        next = q.pop(0)
        visited.append(next)
        child_nodes = graph[next]

        for c in child_nodes:

            if level[c] == -1:
                q.append(c)
                level[c] = level[next]+1

            if level[c] == level[next]+1:
                parents[c].append(next)
                shortestPath[c] = shortestPath[c] + shortestPath[next]

    #print(shortestPath)
    return visited, shortestPath, parents


def find_edge_credits(visited, shortestPath, parents):

    vertex_cred = {v: 1 for v in visited}
    edge_cred = {}

    for v in visited[::-1]:
        for parent in parents[v]:
            edge = min(v, parent), max(v, parent)
            credit = vertex_cred[v] * shortestPath[parent] / shortestPath[v]
            if edge not in edge_cred.keys():
                edge_cred[edge] = 0
            edge_cred[edge] += credit
            vertex_cred[parent] += credit

    return edge_cred


def create_single_community(adjacencyList, vertex):

    q = [vertex]
    community = set()

    while q:
        next = q.pop(0)
        community.add(next)
        for neighbor in adjacencyList[next]:
            if neighbor not in community:
                q.append(neighbor)

    return community


def create_communities(adjacancyList, vertices):

    vertices_1 = vertices.copy()
    result = []
    while vertices_1:
        c = create_single_community(adjacancyList, vertices_1.pop())
        result.append(sorted(list(c)))
        vertices_1 = vertices_1.difference(c)
    return result


def find_modularity(communities, adjacencyList, no_edges):

    modularity = 0.0
    for c in communities:
        for i in c:
            for j in c:
                k_i = len(adjacencyList[i])
                k_j = len(adjacencyList[j])
                a_ij = 1.0 if j in adjacencyList[i] else 0.0
                modularity += a_ij - (k_i * k_j / (2 * no_edges))
    return modularity / (2 * no_edges)

def detect_community(graph, vertices, betweenness_result):

    betweenness_1, adjacencyList = betweenness_result.copy(), graph.copy()
    max_mod = -1
    max_communities = []
    no_edges = len(betweenness_result)

    while betweenness_1:

        communities = create_communities(adjacencyList, vertices)
        modularity = find_modularity(communities, graph, no_edges)
        if modularity > max_mod:
            max_communities = communities
            max_mod = modularity

        max_betweenness = max([v for _, v in betweenness_1])
        discarded_edges = [edge for edge, b_value in betweenness_1 if b_value == max_betweenness]

        #update graph connections
        for e in discarded_edges:
            adjacencyList[e[0]] = {i for i in adjacencyList[e[0]] if i != e[1]}
            adjacencyList[e[1]] = {i for i in adjacencyList[e[1]] if i != e[0]}

        #recalculate betweenness
        betweenness_1 = calc_betweenness(vertices, adjacencyList)

    return max_communities

conf = SparkConf()\
    .setMaster('local[3]')\
    .set("spark.executor.memory", "4g")\
    .set("spark.driver.memory", "4g")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
tick = time.time()

#local run
# threshold = 7
# input_file = "data/HW4/ub_sample_data.csv"
# betweenness_output = "data/HW4/betweenness_output.csv"
# community_output = "data/HW4/community_output.csv"

#terminal run
threshold = int(sys.argv[1])
input_file = sys.argv[2]
betweenness_output = sys.argv[3]
community_output = sys.argv[4]

# read the original json file and remove the header
input_rdd = sc.textFile(input_file)
h = input_rdd.first()

# create map {user_id: [Business_id...]}
UB_map = input_rdd.filter(lambda row: row != h)\
    .map(lambda l: (l.split(',')[0], l.split(',')[1]))\
    .groupByKey()\
    .mapValues(lambda business: sorted(list(business)))\
    .collectAsMap()


# create user pairs
user_pairs = list(combinations(list(UB_map.keys()), 2))
user_pairs_rdd = sc.parallelize(user_pairs)


edges = user_pairs_rdd.map(lambda pair: getEdges(pair)).reduce(add)
edges_rdd = sc.parallelize(edges)

vertices = set(edges_rdd.map(lambda x: x[0]).collect() + edges_rdd.map(lambda x: x[1]).collect())
graph = defaultdict(set)


for edge in edges:
    graph[edge[0]].add(edge[1])
    graph[edge[1]].add(edge[0])

# Determine betweenness values
betweenness_result = calc_betweenness(vertices, graph)
with open(betweenness_output, 'w+') as f:
    for item in betweenness_result:
        f.write(f"{item[0]},{item[1]}\n")
    f.close()

#Find communities
communities = detect_community(graph, vertices, betweenness_result)
communities.sort(key=lambda c: (len(c), c[0]))
with open(community_output, "w+") as f:
    for item in communities:
        f.write("'" + "','".join(item) + "'\n")

tock= time.time()
print("Duration Task 2: ", tock-tick)

