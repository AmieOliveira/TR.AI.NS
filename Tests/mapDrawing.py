#!/usr/local/bin/python3


import csv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Map input
mapPath = "../mapFile"
# ------------------------------

print("Reading map file (%s)" % mapPath)

# Getting CSV file names
graphInfo = "%s/Sheet 1-Graph Info.csv" % mapPath
vertices = "%s/Sheet 1-Vertices Positions.csv" % mapPath
connections = "%s/Sheet 1-Connection Matrix.csv" % mapPath

# Reading Graph Info table
print("\tGoing over graph info")

nVertices = 0
nEdges = 0
with open(graphInfo) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            if not row[0] == "Number of vertices":
                raise ("Wrong input file format. See map input format")
            nVertices = int(row[1])
        else:
            if not row[0] == "Number of connections":
                raise ("Wrong input file format. See map input format")
            nEdges = int(row[1])
        line_count += 1

    print("\t - Map contains %d vertices and %d edges" % (nVertices, nEdges))

# Reading Vertices Positions table
print("\tGoing over vertices positions")

vert_names = []
vert_pos = []
stoppingPoints = {}

with open(vertices) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = -1
    for row in csv_reader:
        if line_count == -1:
            line_count += 1
            continue
        vert_names += [row[0]]
        vert_pos += [(float(row[1]), float(row[2]))]
        if row[0][0] != "_":
           stoppingPoints[row[0]] = line_count
        line_count += 1
    if line_count != nVertices:
        raise ("Wrong input file format. The number of vertices given doesn't match the number of vertices specified")

    print("\t - Got positions of the %d vertices. %d are stopping points" %
              (nVertices, len(stoppingPoints.keys())))

# Reading Connection Matrix table
print("\tGoing over graph edges")

edges = np.ndarray(shape=(nVertices, nVertices), dtype=float)
edges.fill(-1)

with open(connections) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    edge_count = 0
    for row in csv_reader:
        for i in range(nVertices):
            if row[i] != "":
                edges[line_count][i] = float(row[i])
                if line_count > i:
                    edge_count += 1
        line_count += 1
    if nEdges != edge_count:
        raise ("Wrong input file format. Number of edges given doesn't match the specified number")

    print("\t - Read over %d edges in graph" % edge_count)


# Printing map
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

nEdgesDrawn = 0
for i in range(nVertices):
    for j in range(nVertices):
        if j >= i:
            break;
        if edges[i][j] > 0:
            ax.plot( [ vert_pos[i][0], vert_pos[j][0] ], [ vert_pos[i][1], vert_pos[j][1] ], 'k')
            nEdgesDrawn += 1

print(f"{nEdgesDrawn} edges drawn of {nEdges}.")

for ponto in stoppingPoints.keys():
    pos = vert_pos[stoppingPoints[ponto]]
    c = plt.Circle( pos, radius=.4, color='r' )
    ax.add_patch(c)
    ax.text(pos[0]+.1, pos[1]+.2, ponto, fontsize=12, wrap=True)

ax.axes.autoscale()
#plt.show()

import os
import matplotlib.cbook as cbook
import matplotlib.transforms as mtransforms

# Orientacao dada pela velocidade
# Se velocidade for zero, usar o mapa?
path = os.getcwd() + '/../train.png'
with cbook.get_sample_data(path) as image_file:
    image = plt.imread(image_file)

im = ax.imshow(image, extent=[0, 1, 0, 1], clip_on=True)

trans_data = mtransforms.Affine2D().scale(2, 2).rotate_deg(90).translate(0, 5) + ax.transData
im.set_transform(trans_data)

x1, x2, y1, y2 = im.get_extent()
ax.plot(x1, y1,
         transform=trans_data)

plt.show()