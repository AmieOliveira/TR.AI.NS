#!/usr/local/bin/python3

# -----------------------------
#  Simulation code
# TR.AI.NS Project
# -----------------------------

from Train import Train
from Client import Client, CliModes
from Network import Network


import argparse
import csv
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import rcParams
import time

plt.switch_backend('TkAgg')     # Important to make the annimation work!
rcParams['figure.figsize'] = [8, 5.6]

parser = argparse.ArgumentParser(description='Simulation of TR.AI.NS project')

required = parser.add_argument_group('Required Arguments')
required.add_argument( '-m', '--map-file', type=str, required=True,
                       help='Relative path to map files' )

modifiers = parser.add_argument_group('Simulation modifier Arguments')
modifiers.add_argument( '-nT', '--number-of-trains', type=int, default=3,
                        help='Number of trains the simulation should initially contain' )
modifiers.add_argument( '-fC', '--frequency-of-client', type=int, default=25,
                        help='Frequency of clients appearance' )
modifiers.add_argument( '-tS', '--total-steps-run', type=int, default=-1,
                        help='Total number of steps one wishes the simulation to run.'
                             'If nothing is specified, simulation will stop after'
                             'delivering 10 clients.' )
modifiers.add_argument( '-vS', '--step-speed', type=float, default=1,
                        help='Ratio of of steps per second' )
modifiers.add_argument( '-cR', '--client-range', type=float, default=.5,
                        help='This parameter is a number between 0 and 1, and sets '
                             'the client broadcast radius to a fraction of the map '
                             'diagonal.' )


args = parser.parse_args()

client_range = args.client_range
if client_range <= 0:
    raise Exception("ERROR: Wrongly specified argument! CLIENT-RANGE should be bigger than zero")
elif client_range > 1:
    print("WARNING: CLIENT-RANGE argument bigger than 1. Automatically set to 1.")
    client_range = 1


# TODO: Create interface with user
#   Ideally it should allow is to:
#     - Create a client at will (selecting its position and destination or leaving it to chance)
#     - Add trains or remove selected ones
#     - Put trains in outOfOrder to move them to designed places
#     - Any other cool ideas


class Simulation:
    def __init__(self):
        self.devices = []
        self.clientRange = 40
        self.trainRange = 120

running = 1       # Global variable to say if simulation should be running

class Index(object):
    run = 1

    def pause_play(self, event):
        print("Button pressed!")
        global running

        if self.run == 0:
            self.run = 1
            running = 1
            button.label.set_text("Pause")
        elif self.run == 1:
            self.run = 0
            running = 0
            button.label.set_text("Play")

            print( "Simulation paused" )

    def statistics(self, event, waitingTime, devices):
        print("Button pressed")

        outText = ""

        if len(waitingTime.keys()) != 0:
            n = 0
            total = 0
            for key in waitingTime.keys():
                n += 1
                total += waitingTime[key][1]
            meanWaitTime = total/n

            outText += "Average total waiting time is of {:.1f} simulation counts.\n{} clients delivered.\n\n".\
                format(meanWaitTime, n)

        sumDistance = 0
        for dev in devices:
            if isinstance(dev, Train):
                sumDistance += dev.totalDistanceRun
        outText += "Total distance run by all trains: {} m".format(sumDistance)

        figW = plt.figure(10, figsize=(5, 2))
        figW.text(.1, .5, outText)



callback = Index()

# Main funtion
if __name__ == "__main__":
    # Loading map:
    mapPath = args.map_file
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

    # Availability dictionary
    availability = {}

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

    # ------------------------------
    # Creating Network
    sim = Simulation()

    net = Network(sim, log=False)

    sim.clientRange = int(map_size * client_range)
    sim.trainRange = 3 * sim.clientRange

    # ------------------------------
    # Creating train objects
    nTrains = args.number_of_trains

    v_step = args.step_speed

    for i in range(nTrains):
        pos = vert_pos[ randint(0,nVertices-1) ]
        tr = Train(i, pos, v_step, mapPath, availability, net, log=True)
        sim.devices += [tr]

    # ------------------------------
    # Creating initial client object
    nClients = 0

    currCli = 0.5
    clientList = []

    init = randint(0, len(stoppingPointsPos) - 1)
    fin = randint(0, len(stoppingPointsPos) - 1)
    if fin == init:
        fin += 1
        if fin == len(stoppingPointsPos):
            fin = 0

    pos = stoppingPointsPos[ init ]
    dest = stoppingPointsPos[ fin ]

    cl = Client(currCli, pos, dest, v_step, mapPath, net, log=True)
    sim.devices += [cl]

    clientList += [cl]
    outingClients = {}

    # ------------------------------
    # Creating waiting time data
    waitingTime = {}

    # ------------------------------
    # Looping simulation
    finished = False
    simTime = 0

    out_file = open("log.txt", "w")

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle( "TR.AI.NS Simulation", fontweight='bold', fontsize=17 )

    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    plt.show(block=False)

    bax = plt.axes([0.05, 0.01, 0.1, 0.075])
    button = Button(bax, 'Pause')
    bSax = plt.axes([0.16, 0.01, 0.1, 0.075])
    buttonS = Button(bSax, 'Stats')

    while not finished:
        if running:
            clockcount = simTime * v_step
            print( "Simulation counter: {}".format(clockcount) )

            r = randint(1, 100)
            if r % args.frequency_of_client == 0:
                currCli += 1

                init = randint(0, len(stoppingPointsPos) - 1)
                fin = randint(0, len(stoppingPointsPos) - 1)
                if fin == init:
                    fin += 1
                    if fin == len(stoppingPointsPos):
                        fin = 0

                pos = stoppingPointsPos[init]
                dest = stoppingPointsPos[fin]

                cl = Client(currCli, pos, dest, v_step, mapPath, net, log=True)
                sim.devices += [cl]
                clientList += [cl]

            # Run all devices
            for device in sim.devices:
                device.step()
        else:
            pass

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

        # Remove clients from list
        for client in clientList:
            if client.mode == CliModes.dropoff:
                if client not in outingClients.keys():
                    outingClients[client] = 0
                    waitingTime[client.id] = (client.timeTillRequest, client.waitingTime)
                else:
                    outingClients[client] += 1

                if outingClients[client] >= 10:
                    # Removing client from simulation
                    sim.devices.remove(client)
                    clientList.remove(client)

                    client.kill()
                    nClients += 1

        out_file.write( "Simulation step {}, timer {}\n".format(simTime, simTime*v_step) )
        for device in sim.devices:
            out_file.write( "\tDevice {}, mode {}: position {}\n".format(device.id, device.mode, device.pos) )

            if isinstance(device, Train):
                out_file.write( "\t  Processing request {}\n".format(device.unprocessedReqs) )
                out_file.write( "\t  Path {}\n".format(device.path) )
                out_file.write( "\t  Clients list {}\n".format(device.client) )
                out_file.write( "\t  Clients en route {}\n".format(device.inCourseClients) )
            elif isinstance(device, Client):
                out_file.write( "\t  Destination: {}.\n".format(device.destiny) )
                out_file.write( "\t  Train that will pick me up {}\n".format(device.train) )
        out_file.write("\n")

        if running:
            simTime += 1

        button.on_clicked(callback.pause_play)
        buttonS.on_clicked(lambda x: callback.statistics(x, waitingTime, sim.devices))

        # TODO: Understando why buttons are clicked several times, and correct it

        if args.total_steps_run != -1:
            if simTime >= args.total_steps_run:
                finished = True
        elif nClients >= 15:
            finished = True

    out_file.close()
    print("Finished simulation!")
    plt.show()