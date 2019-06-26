#!/usr/bin/env python
# coding: utf-8

# In[47]:


#pip install networkx
from scipy.spatial import distance


# In[15]:


# coding: utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
import itertools

edgelist = pd.read_csv('mapFile//edgeList.csv')
nodelist = pd.read_csv('mapFile//Sheet 1-Vertices Positions.csv')
#print(edgelist)
nodelist.head()
# Create empty graph
myGraph = nx.Graph()
# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    myGraph.add_edge(elrow[0], elrow[1], distance=elrow[2])
for i, nlrow in nodelist.iterrows():
    #myPathGraph.node[nlrow[0]] = nlrow[1:].to_dict()
    myGraph.node[nlrow[0]].update(nlrow[1:].to_dict())

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
    paths = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1],measure)
                paths[pair] = nx.dijkstra_path(graph, pair[0], pair[1],measure)
    return distances,paths

my_nodes = list(myGraph.nodes)
# Compute all pairs of odd nodes. in a list of tuples
my_node_pairs = list(itertools.combinations(my_nodes, 2))

# Compute shortest paths.  Return a dictionary with node pairs keys and a single value equal to shortest path distance.
node_pairs_shortest_paths,paths = get_shortest_paths_distances(myGraph, my_node_pairs, 'distance')

print(node_pairs_shortest_paths)


# In[28]:


test = [('_1', 'Point_3')]


# In[30]:


result = get_shortest_paths_distances(myGraph, test, 'distance')
print(result)


# In[35]:


def calculate_route(graph, init, fin, measure="distance"):
    points_to_calculate = [(init, fin)]
    distances_length = nx.dijkstra_path_length(graph, points_to_calculate[0][0], points_to_calculate[0][1], measure)
    distances_path = nx.dijkstra_path(graph, points_to_calculate[0][0], points_to_calculate[0][1], measure)
    return distances_path,distances_length


# In[37]:


result = calculate_route(myGraph, "_1","Point_3", 'distance')
print(result)


# In[39]:


def calculate_route2(graph, init, fin, measure="distance"):
    #points_to_calculate = [(init, fin)]
    distances_length = nx.dijkstra_path_length(graph,init,fin,measure)
    distances_path = nx.dijkstra_path(graph,init,fin,measure)
    return distances_path,distances_length


# In[40]:


result = calculate_route2(myGraph, "_1","Point_3", 'distance')
print(result)


# In[41]:


calculate_route2(myGraph, "_1","Point_3", 'distance')


# In[45]:


nodelist1 = pd.read_csv('mapFile1//Sheet 1-Vertices Positions.csv',delimiter=";")


# In[46]:


nodelist1


# In[99]:


def discover_proximity_point(init=0, fin=0):
    dist ={}
    minVal = 1000
    for index,noderow in nodelist1.iterrows():
        #dist.append(distance.euclidean((int(noderow[1]),int(noderow[2])),(init,fin)))
        value = distance.euclidean((int(noderow[1]),int(noderow[2])),(init,fin))
        dist[(noderow[1],noderow[2])] = value
        if(value<minVal):
            minVal = value
            init_temp,fin_temp = noderow[1],noderow[2]
    print(min(dist.values()))
    if (min(dist.values())== 0):
        return (init,fin)
    else:
        return (init_temp,fin_temp)


# In[103]:


discover_proximity_point(20,30)


# In[ ]:





# In[ ]:




