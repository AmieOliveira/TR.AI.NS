#!/usr/local/bin/python3

# -----------------------------
#  Simulation code
# TR.AI.NS Project
# -----------------------------

from Train import Train
from Client import Client
from Protocol import Message, MsgTypes
from Network import Network


import argparse
import csv
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import time

plt.switch_backend('TkAgg')

parser = argparse.ArgumentParser(description='Simulation of TR.AI.NS project')

required = parser.add_argument_group('Required Arguments')
required.add_argument( '-m', '--mapFile', type=str,required=True,
                       help='Relative path to map files' )


args = parser.parse_args()


class Simulation:
    def __init__(self):
        self.devices = []
        self.trainRange = 15
        self.clientRange = 5


# Main funtion
if __name__ == "__main__":
    # Loading map:
    mapPath = args.mapFile
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
            raise (
            "Wrong input file format. The number of vertices given doesn't match the number of vertices specified")

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


    # ------------------------------
    # Creating Network
    sim = Simulation()

    net = Network(sim, log=True)

    # ------------------------------
    # Creating train objects
    nTrains = 3
    # TODO: Put as an argument...

    for i in range(nTrains):
        pos = vert_pos[ randint(0,nVertices-1) ]
        tr = Train(i, pos, mapFile=mapPath, log=True)
        sim.devices += [tr]

    # ------------------------------
    # Creating initial client object
    pos = vert_pos[ stoppingPoints["Point 1"] ]
    cl = Client(.5, pos, mapPath, log=True)
    sim.devices += [cl]

    # ------------------------------
    # Looping simulation
    finished = False
    simTime = 0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.show(block=False)

    while not finished:
        simTime += 1
        print(simTime)

        # Run all devices
        for device in sim.devices:
            device.step()

        # Print map
        nEdgesDrawn = 0
        for i in range(nVertices):
            for j in range(nVertices):
                if j >= i:
                    break;
                if edges[i][j] > 0:
                    ax.plot([vert_pos[i][0], vert_pos[j][0]], [vert_pos[i][1], vert_pos[j][1]], 'k')
                    nEdgesDrawn += 1
        # print(f"{nEdgesDrawn} edges drawn of {nEdges}.")

        for ponto in stoppingPoints.keys():
            pos = vert_pos[stoppingPoints[ponto]]
            c = plt.Circle(pos, radius=.4, color='r')
            ax.add_patch(c)
            ax.text(pos[0] + .1, pos[1] + .2, ponto, fontsize=12, wrap=True)

        ax.axes.autoscale()

        for device in sim.devices:
            device.draw(ax)

        plt.show(block=False)
        fig.canvas.flush_events()
        time.sleep(.5)

        if simTime >= 5:
            finished = True

    plt.show()