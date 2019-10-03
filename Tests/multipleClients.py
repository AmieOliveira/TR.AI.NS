#!/usr/local/bin/python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from Train import Train
from Client import Client, CliModes
from Network import Network

import csv
import numpy as np
from time import sleep
import matplotlib.pyplot as plt


plt.switch_backend('TkAgg')

class Simulation:
    def __init__(self):
        self.devices = []
        self.trainRange = 0
        self.clientRange = 0

# Reading Map
mapPath = "../map_grid"
print("Reading map file (%s)" % mapPath)

# Getting CSV file names
graphInfo = "%s/Sheet 1-Graph Info.csv" % mapPath
vertices = "%s/Sheet 1-Vertices Positions.csv" % mapPath
connections = "%s/Sheet 1-Connection Matrix.csv" % mapPath

# Reading Graph Info table
print("\tGoing over graph info")
nVertices = 0
nEdges = 0
map_size = 0
with open(graphInfo) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            if not row[0] == "Number of vertices":
                raise Exception("Wrong input file format. See map input format")
            nVertices = int(row[1])
        elif line_count == 1:
            if not row[0] == "Number of connections":
                raise Exception("Wrong input file format. See map input format")
            nEdges = int(row[1])
        else:
            if not row[0] == "Map size":
                raise Exception("Wrong input file format. See map input format")
            map_size = float(row[1])
        line_count += 1
    print("\t - Map contains %d vertices and %d edges" % (nVertices, nEdges))

# Reading Vertices Positions table
print("\tGoing over vertices positions")
vert_pos = []
stoppingPoints = {}
stoppingPointsPos = []
with open(vertices) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = -1
    for row in csv_reader:
        if line_count == -1:
            line_count += 1
            continue
        vert_pos += [(float(row[1]), float(row[2]))]
        if row[0][0] != "_":
            stoppingPoints[row[0]] = line_count
            stoppingPointsPos += [(float(row[1]), float(row[2]))]
        line_count += 1
    if line_count != nVertices:
        raise Exception("Wrong input file format. The number of vertices given doesn't match the number of vertices specified")
    print("\t - Got positions of the %d vertices. %d are stopping points" %
          (nVertices, len(stoppingPoints.keys())))

# Reading Connection Matrix table
print("\tGoing over graph edges")
edges = np.ndarray(shape=(nVertices, nVertices), dtype=float)
edges.fill(-1)
availability = {} # Availability dictionary
with open(connections) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    edge_count = 0
    for row in csv_reader:
        for i in range(nVertices):
            if row[i] != "":
                edges[line_count][i] = float(row[i])
                if line_count != i:
                    a = max(line_count, i)
                    b = min(line_count, i)
                    availability[ (a, b) ] = True
                if line_count > i:
                    edge_count += 1
        line_count += 1
    if nEdges != edge_count:
        raise Exception("Wrong input file format. Number of edges given doesn't match the specified number")
    print("\t - Read over %d edges in graph" % edge_count)


# Simulation variables
sim = Simulation()
net = Network(sim, log=False)
sim.clientRange = int(map_size * .5)
sim.trainRange = 3 * sim.clientRange

vStep = 2

# Creating train object
trainPosition = (0, 3000)
tr = Train(1, trainPosition, vStep, mapPath, availability, net, log=True)
sim.devices += [tr]


# Creating client objects
c1info = {
    'id':0.5,
    'time':10,
    'pos':(2000,3500),
    'dest':(2000,0)
} # First client starts in t=10s, in Pinheiro and desires to go to Dantas
c2info = {
    'id':1.5,
    'time':70,
    'pos':(2000,2000),
    'dest':(4000,1000)
} # Second client starts at t=70s, in Pau-ferro, and desires to go to Jacarepagua


# Looping simulation
finished = False
simTime = 0

fig = plt.figure(figsize=(10, 10))
fig.suptitle( "TR.AI.NS Simulation", fontweight='bold', fontsize=17 )
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
plt.show(block=False)

while not finished:
    clockcount = simTime * vStep
    print( "Simulation counter: {}".format(clockcount) )

    if clockcount == c1info['time']:
        cl1 = Client(c1info['id'], c1info['pos'], c1info['dest'], vStep, mapPath, net, log=True)
        sim.devices += [cl1]
    elif clockcount == c2info['time']:
        cl2 = Client(c2info['id'], c2info['pos'], c2info['dest'], vStep, mapPath, net, log=True)
        sim.devices += [cl2]


    # Run all devices
    for device in sim.devices:
        device.step()

    # Print map
    ax.cla()
    nEdgesDrawn = 0
    for i in range(nVertices):
        for j in range(nVertices):
            if j >= i:
                break
            if edges[i][j] > 0:
                ax.plot([vert_pos[i][0], vert_pos[j][0]], [vert_pos[i][1], vert_pos[j][1]], 'k', zorder=-4)
                nEdgesDrawn += 1
    # print(f"{nEdgesDrawn} edges drawn of {nEdges}.")
    xmin, xmax, ymin, ymax = ax.axis()
    scale = (ymax-ymin) * .016  # Scale fator to print visible circles
    for ponto in stoppingPoints.keys():
        pos = vert_pos[stoppingPoints[ponto]]
        c = plt.Circle(pos, radius=scale, color='r', zorder=-5)
        ax.add_patch(c)
        ax.text(pos[0] + scale*.5, pos[1] + scale, ponto, fontsize=12, wrap=True, zorder=-3)
    xmin, xmax, ymin, ymax = ax.axis()
    diverge = .05
    xmin = xmin - (xmax - xmin) * diverge
    xmax = xmax + (xmax - xmin) * diverge
    ymin = ymin - (ymax - ymin) * diverge
    ymax = ymax + (ymax - ymin) * diverge
    ax.axis([xmin, xmax, ymin, ymax])

    for device in sim.devices:
        device.draw(ax)

    clockcount = float(clockcount)
    hour = int(clockcount // 3600)
    clockcount %= 3600
    minutes = int(clockcount // 60)
    clockcount %= 60
    seconds = int (clockcount)
    ax.text(0.95, -0.1, 'Time {:02d}:{:02d}:{:02d}'.format(hour, minutes, seconds),
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=15)
    plt.show(block=False)       # The False argument makes the code keep running even if I don't close the plot window
    fig.canvas.flush_events()

    simTime += 1

    try:
        if cl1.mode == CliModes.dropoff and cl2.mode == CliModes.dropoff:
            finished = True
    except:
        pass

print("Finished simulation!")
plt.show()