#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
G.add_weighted_edges_from(e)
print(nx.dijkstra_path(G, 'a', 'd'))

plt.figure(0)
nx.draw(G)