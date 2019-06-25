# coding: utf-8
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
import itertools

edgelist = pd.read_csv('mapFile/edgeList.csv')
nodelist = pd.read_csv('mapFile/Sheet 1-Vertices Positions.csv')
#print(edgelist)
nodelist.head()
# Create empty graph
myGraph = nx.Graph()
# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    myGraph.add_edge(elrow[0], elrow[1], distance=elrow[2])
for i, nlrow in nodelist.iterrows():
    #myPathGraph.node[nlrow[0]] = nlrow[1:].to_dict()
    myGraph.node[nlrow['Name']].update(nlrow[1:].to_dict())

#list(myGraph.edges(data=True))
#list(myGraph.nodes(data=True))

# Define node positions data structure (dict) for plotting
node_positions = {node[0]: (node[1]['X'], node[1]['Y']) for node in myGraph.nodes(data=True)}

plt.figure(figsize=(8, 6))
nx.draw(myGraph, pos=node_positions,node_size=10, node_color='black',with_labels = True)
plt.title('Graph Representation of Train Map', size=15)
plt.show()

def get_shortest_paths_distances(graph, pairs, measure):
    """
    Compute shortest distance between each pair of nodes in a graph.  Return a dictionary keyed on node pairs (tuples).
    """
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1],measure)
    return distances

my_nodes = list(myGraph.nodes)
# Compute all pairs of odd nodes. in a list of tuples
my_node_pairs = list(itertools.combinations(my_nodes, 2))

# Compute shortest paths.  Return a dictionary with node pairs keys and a single value equal to shortest path distance.
node_pairs_shortest_paths = get_shortest_paths_distances(myGraph, my_node_pairs, 'distance')

print(node_pairs_shortest_paths)



