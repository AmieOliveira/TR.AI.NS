"""
    Train Robot Classes
    TR.AI.NS Project
    Authors: Amanda, Vinicius
"""

__all__ = ['Train']
from Protocol import Message, MsgTypes
from enum import Enum
from random import randint
import csv
import numpy as np
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.transforms as mtransforms
import networkx as nx
from math import atan2


class TrainModes(Enum):
    """
        Group of possible train operational states at any given moment
    'wait' -> No clients
    'accept' -> Going to pick up client
    'busy' -> Taking client to dropOff location
    'outOfOrder' -> Moving due to system order
    """
    wait = 1
    accept = 2
    busy = 3
    outOfOrder = 4


class Train:
    def __init__(self, ID, pos0, vStep, mapFile, edgeAvaliability, network, vMax = 20, log=False):
        """
            Class Train contains the whole operational code for a transportation unit in
            the TR.AI.NS project.
        :param ID: Gives the ID number of the train. Every train should have a distinct ID
          (which should also be different from the client IDs).
          Norm says trains should have integer identifiers
        :param pos0: Initial position of the train. Format should be '(x, y)', and needs to
          be either vertice or edge on map
        :param vStep: Speed of time passage. Gives the ratio of seconds per step
        :param mapFile: Name of the file that contains the map information
        :edgeAvaliability: Shared object amongst trains that simulates a semaphore of sorts
          in the simulation, preventing collisions
        :network: Object that simulates a broadcast medium of communication
        :vMax: Train maximum speed. As the code stands, it is the speed the train always
         applies when moving
        :log: Boolean parameter that enables or disables train's prints
        """
        self.id = ID

        # Network object
        self.network = network

        # Logging variable
        self.log = log

        # Moving attributes
        self.pos = tuple(pos0)                 # Current position of the train

        self.currentEdge = None

        self.vStep = vStep                  # approximate s/step ratio
        self.v = [0, 0]                      # train speed in m/s
            # Quando evoluir adiconar aceleração

        self.vMax = vMax                 # Maximum train speed in m/s
        #self.aMax = 4                   # Maximum train acceleration in m/s^2
        # TODO: Add acceleration to move method

        self.okToMove = True
        self.waitForClientDelay = 0
        self.nominalClientWaitingTime = 10 # In seconds. Time a train should wait for boarding and departure of clients
        self.clientWaitingTime = self.nominalClientWaitingTime / self.vStep # Converted to number of steps

        # Messaging attributes
        self.messageBuffer = []

        # Map attributes
        self.load_map(mapFile)
        self.semaphore = edgeAvaliability

        # Operational attributes    (related to the paths being and to be taken)
        self.mode = TrainModes.wait        # Current mode of operation of the train

        self.inCourseClients = []
        self.client = {}                # List of pickup and dropOff locations for the next clients, with the client ID
                                        # [(Id1, pickup1, dropoff1), ...]
        self.notifiedClients = False

        self.path = []                 # List of vertices '(x, y)' to be passed through
        self.goals = []          # List which contains where clients board and leave train
        self.currentGoal = None

        # Statistics
        self.totalDistanceRun = 0

        # Elections variables
        self.unprocessedReqs = {}       # Client request that is on process of train elections
                                        # Requests handled in dictionaries. ONLY ONE ALLOWED PER TURN
        self.outOfElec = None           # There has been a client request and this is not the elected train

        self.nominalDelayWanted = randint(1,10) # In seconds. Delay to send the election message
        self.delayWanted = self.nominalDelayWanted / self.vStep # Converted to number of steps

        self.nominalMaximumMsgWait = 40 # In seconds. Time a train should wait for answer from other trains before
                                        # declaring himself winner of the election process.
        self.maximumMsgWait = self.nominalMaximumMsgWait / self.vStep # converted to number of steps.
        # ATTENTION! DUE TO THE WAY THE SIMULATION IS IMPLEMENTED, ONE CANNOT AUGMENT TOO MUCH THE STEP SPEED!
        # There could be information loss

        # Train gif image
        self.img = os.path.dirname(os.path.abspath(__file__)) + '/train.png'
    # -----------------------------------------------------------------------------------------

    def step(self):
        """
            This method executes the whole operation of the robot during a logic step.
            Should be looped to have the robot functioning.
        """
        # Time counting updates
        if 'ID' in self.unprocessedReqs.keys():
            if not self.unprocessedReqs['inElections']:
                self.unprocessedReqs['delayT'] += 1
            else:
                self.unprocessedReqs['msgWait'] += 1
        if not self.okToMove:
            self.waitForClientDelay += 1

        # Reading and interpreting messages in the message buffer
        currentMessage = None
        if len(self.messageBuffer) > 0:
            # In this case there are messages to be interpreted
            currentMessage = self.messageBuffer.pop(0)

        if currentMessage:
            if self.log:
                print(" \033[94mTrain {}:\033[0m Received message '{}'".format(self.id, currentMessage.nType.name))
                # print "\t %s" % str(currentMessage.msgDict)

            # Case 1: Service request from client
            if currentMessage['type'] == MsgTypes.req.value:

                if self.mode != TrainModes.outOfOrder: # Checks if train can accept
                    if not ('ID' in self.unprocessedReqs.keys()): # Checks if there are current processes ongoing
                        clientID = currentMessage['sender']

                        if self.log:
                            print(" \033[94mTrain {}:\033[0m Processing Client {} Request".format(self.id, clientID))

                        route, goals, d = None, None, None
                        # Calculate route

                        # First check if the goal is on path
                        pickup = tuple(currentMessage['pickUp'])
                        dropoff = tuple(currentMessage['dropOff'])
                        foundViable = False

                        if pickup in self.path:
                            idx = self.path.index(pickup)
                            d_to_pu = self.full_distance(stop=idx)
                            timeTillArrival = d_to_pu / self.vStep

                            if timeTillArrival >= 1.5*self.nominalMaximumMsgWait:
                                foundViable = True
                                # Client is in path, will run the election considering this
                                if dropoff in self.path[idx:]:
                                    # Drop-off position is also in the path
                                    d_idx = self.path[idx:].index(dropoff) + idx
                                    d = d_to_pu + self.full_distance(start=idx,stop=d_idx)

                                    # Begin elections immediately (time sensitive case)
                                    self.unprocessedReqs = dict(ID=clientID, pickup=pickup, dropoff=dropoff,
                                                                inElections=True, d=d, msgWait=0,
                                                                pick_idx=idx, drop_idx=d_idx)
                                    # NOTE: The indexes need to be updated when a vertice is taken out of the path!
                                    self.start_election(d)
                                    if self.log:
                                        print( " \033[94mTrain {}:\033[0m Starting Election!".format(self.id) )

                                else:
                                    # Need to calculate the route to drop-off
                                    route, d_drop = self.calculate_route( self.path[-1], dropoff )
                                    route = route[1:]
                                    goals = [None]*len(route)
                                    goals[-1] = [clientID]

                                    d = self.full_distance() + d_drop

                                    # Begin elections immediately (time sensitive case)
                                    self.unprocessedReqs = dict(ID=clientID, pickup=pickup, dropoff=dropoff,
                                                                inElections=True, d=d, msgWait=0,
                                                                pick_idx=idx, goalList=goals, route=route)
                                    self.start_election(d)
                                    if self.log:
                                        print(" \033[94mTrain {}:\033[0m Starting Election!".format(self.id))

                            else:
                                auxPath = self.path[idx:]

                                while pickup in auxPath:
                                    idx = auxPath.index(pickup)
                                    d_to_pu = self.full_distance(stop=idx)
                                    timeTillArrival = d_to_pu / self.vStep

                                    if timeTillArrival >= 1.5 * self.nominalMaximumMsgWait:
                                        foundViable = True
                                        # Client is in path, will run the election considering this
                                        if dropoff in self.path[idx:]:
                                            # Drop-off position is also in the path
                                            d_idx = self.path[idx:].index(dropoff) + idx
                                            d = d_to_pu + self.full_distance(start=idx, stop=d_idx)

                                            # Begin elections immediately (time sensitive case)
                                            self.unprocessedReqs = dict(ID=clientID, pickup=pickup, dropoff=dropoff,
                                                                        inElections=True, d=d, msgWait=0,
                                                                        pick_idx=idx, drop_idx=d_idx)
                                            # NOTE: The indexes need to be updated when a vertice is taken out of the path!
                                            self.start_election(d)
                                            if self.log:
                                                print(" \033[94mTrain {}:\033[0m Starting Election!".format(self.id))

                                        else:
                                            # Need to calculate the route to drop-off
                                            route, d_drop = self.calculate_route(self.path[-1], dropoff)
                                            route = route[1:]
                                            goals = [None] * len(route)
                                            goals[-1] = [clientID]

                                            d = self.full_distance() + d_drop

                                            # Begin elections immediately (time sensitive case)
                                            self.unprocessedReqs = dict(ID=clientID, pickup=pickup, dropoff=dropoff,
                                                                        inElections=True, d=d, msgWait=0,
                                                                        pick_idx=idx, goalList=goals, route=route)
                                            self.start_election(d)
                                            if self.log:
                                                print(" \033[94mTrain {}:\033[0m Starting Election!".format(self.id))

                                        break
                                    auxPath = auxPath[idx:]

                        if not foundViable:
                            # Will proceed to calculate route from end of path
                            if self.path == []:
                                # There is no current path route
                                pick_route, pick_d = self.calculate_route( self.pos, pickup )
                            else:
                                pick_route, pick_d = self.calculate_route( self.path[-1], pickup )
                                pick_route = pick_route[1:]

                            drop_route, drop_d = self.calculate_route( pickup, dropoff )
                            drop_route = drop_route[1:]

                            route = pick_route + drop_route
                            d = pick_d + drop_d

                            # NOTE: To differentiate the pickup from the dropoff, it was decided to put the negative
                            # of the client ID when the goal corresponds to a pickup
                            pick_goals = [None]*(len(pick_route)-1) + [[-clientID]]
                            drop_goals = [None]*(len(drop_route)-1) + [[clientID]]
                            goals = pick_goals + drop_goals

                            self.unprocessedReqs = dict(ID=clientID, pickup=pickup, dropoff=dropoff,
                                                        inElections=False, delayT=0, d=d, msgWait=0,
                                                        simpleD=pick_route, route=route, goalList=goals)

                        self.acknowlege_request()
                        # Create a message type to indicate to client that the request has been heard and is being processed

            # Case 2: Election started
            elif currentMessage['type'] == MsgTypes.elec.value:

                if not self.mode == TrainModes.outOfOrder:  # Checks if train can accept
                    # if not self.outOfElec == currentMessage['clientID']: # Check if has already 'lost' election

                    if 'ID' in self.unprocessedReqs.keys():
                        if self.unprocessedReqs['ID'] == currentMessage['clientID']:
                            # NOTE: I assume any car receives first the notice from the client
                            if self.log:
                                print(" \033[94mTrain {}:\033[0m Received Election Message (from {}, d={})".
                                      format(self.id, currentMessage['sender'], currentMessage['distance']))

                            dTot = self.unprocessedReqs['d']

                            if (dTot < currentMessage['distance']) or \
                                    (dTot == currentMessage['distance'] and self.id > currentMessage['sender']):
                                # This train is the leader himself
                                self.silence_train(currentMessage['sender'])
                                if not self.unprocessedReqs['inElections']:
                                    # If It hasn't yet send its distance, should do so now
                                    self.start_election(dTot)
                                    self.unprocessedReqs['inElections'] = True
                                    self.unprocessedReqs['msgWait'] = 0

                                if self.log:
                                    print( " \033[94mTrain {}:\033[0m Win this elections round".format(self.id) )

                            else:
                                # Finishes current election process
                                self.outOfElec = self.unprocessedReqs['ID']
                                self.unprocessedReqs = {}

                                if self.log:
                                    print( " \033[94mTrain {}:\033[0m Lost these elections".format(self.id) )

            # Case 3: Election answer
            elif currentMessage['type'] == MsgTypes.elec_ack.value:
                if "ID" in self.unprocessedReqs.keys():
                    if self.unprocessedReqs['ID'] == currentMessage['clientID']: # Checks if this message is from current election
                        # No need to check if message is destined to itself, because the receiving mechanism already does so.
                        # Train lost current election. Finishes election process
                        self.outOfElec = self.unprocessedReqs['ID']
                        self.unprocessedReqs = {}

                        if self.log:
                            print( " \033[94mTrain {}:\033[0m Silenced in these elections. Lost election.".format(self.id) )

            # Case 4: Leader Message
            elif currentMessage['type'] == MsgTypes.leader:
                if "ID" in self.unprocessedReqs.keys():
                    if self.unprocessedReqs['ID'] == currentMessage['clientID']: # Checks if this message is from current election
                        self.outOfElec = self.unprocessedReqs['ID']
                        self.unprocessedReqs = {}

                        if self.log:
                            print( " \033[94mTrain {}:\033[0m Got an election leader in these elections. Lost election.".format(self.id) )

            # Any other type of message is certainly not destined to myself, therefore no action is taken
            else:
                pass
        # ------------------------------------------

        # Election start
        if 'ID' in self.unprocessedReqs.keys():
            if not self.unprocessedReqs['inElections']:
                if self.unprocessedReqs['delayT'] >= self.delayWanted:
                    # Will start election
                    if self.log:
                        print( " \033[94mTrain {}:\033[0m Starting Election!".format(self.id) )

                    self.unprocessedReqs['inElections'] = True
                    d = self.unprocessedReqs['d']
                    self.start_election(d)
                    self.unprocessedReqs['msgWait'] = 0
        # ------------------------------------------

        # Elections finish
            else:
                if self.unprocessedReqs['msgWait'] >= self.maximumMsgWait:
                    # If no answer is given, election isn't silenced and I am current leader
                    # self.broadcast_leader(self.id) # Inform others who's answering the request

                    if self.log:
                        print( " \033[94mTrain {}:\033[0m Finishing election! I've won!".format(self.id) )

                    if 'route' in self.unprocessedReqs.keys():
                        self.path += self.unprocessedReqs['route'] # Adds route to desired path
                        self.goals += self.unprocessedReqs['goalList']

                    if 'pick_idx' in self.unprocessedReqs.keys():
                        if not self.goals[ self.unprocessedReqs['pick_idx'] ]:
                            self.goals[ self.unprocessedReqs['pick_idx'] ] = [-self.unprocessedReqs['ID']]
                        else:
                            self.goals[ self.unprocessedReqs['pick_idx'] ] += [-self.unprocessedReqs['ID']]

                        if 'drop_idx' in self.unprocessedReqs.keys():
                            if not self.goals[self.unprocessedReqs['drop_idx']]:
                                self.goals[ self.unprocessedReqs['drop_idx'] ] = [self.unprocessedReqs['ID']]
                            else:
                                self.goals[self.unprocessedReqs['drop_idx']] += [self.unprocessedReqs['ID']]

                    elif 'simpleD' in self.unprocessedReqs.keys():
                        if self.unprocessedReqs['simpleD'] == 0 and self.mode == TrainModes.wait:
                            self.okToMove = False
                            self.waitForClientDelay = 0

                    # In this case I'd need to convert into coordinates
                    self.client[self.unprocessedReqs['ID']] = [(self.unprocessedReqs['pickup'], self.unprocessedReqs['dropoff'])]
                    self.client_accept()
                    self.unprocessedReqs = {} # Finishes current election process

                    if self.mode == TrainModes.wait:
                        self.mode = TrainModes.accept
        # ------------------------------------------

        # Moving train and handling new position
        if self.currentGoal and (not self.okToMove):
            for client in self.currentGoal:
                action = "board" if (client < 0) else "disembark"
                clientID =abs(client)

                if self.log:
                    print(" \033[94mTrain {}:\033[0m Waiting for client to {} ({})".
                        format(self.id, action, clientID))

            if self.waitForClientDelay >= self.clientWaitingTime:
                self.okToMove = True
                self.currentGoal = None
                self.notifiedClients = False

        self.move()

        # TODO
        if self.currentGoal and (not self.notifiedClients):
            # Reached pick-up or drop-off destination
            # Proceding to notify clients

            for client in self.currentGoal:
                self.notify_client(client)

                if client < 0:
                    # Boarding train
                    self.inCourseClients.append(abs(client))
                    if self.mode == TrainModes.accept:
                        self.mode = TrainModes.busy

                elif client > 0:
                    # Leaving train
                    self.inCourseClients.remove(abs(client))
                    del self.client[abs(client)]
                    if self.mode == TrainModes.busy and (not self.inCourseClients):
                        if len(self.client) > 0:
                            self.mode = TrainModes.accept
                        else:
                            self.mode = TrainModes.wait

            self.notifiedClients = True
    # -----------------------------------------------------------------------------------------

    def receive_message(self, msgStr):
        """
            Receives message in string format and converts it into a protocol class
        :param msgStr: Should be a string coded with the message
        """
        msg = Message()
        msg.decode(msgStr)

        if msg['sender'] == self.id:
            return

        if msg.nType in [MsgTypes.req_ack, MsgTypes.req_ans]:
            return

        elif msg.nType in [MsgTypes.req, MsgTypes.elec, MsgTypes.leader]:
            self.messageBuffer += [msg]

        else:
            if msg['receiver'] == self.id:
                self.messageBuffer += [msg]
    # -----------------------------------------------------------------------------------------

    def load_map(self, mapPath):
        """
            Loads map information into the train object. Sets up necessary attributes
        :param mapPath: The folder path for the CSV files with the map content. Files
            must be created according to the model file format
        """

        if self.log:
            print(" \033[94mTrain {}:\033[0m Reading map file ({})".format(self.id, mapPath))

        # Getting CSV file names
        graphInfo = "%s/Sheet 1-Graph Info.csv" % mapPath
        vertices = "%s/Sheet 1-Vertices Positions.csv" % mapPath
        connections = "%s/Sheet 1-Connection Matrix.csv" % mapPath

        self.graph = nx.Graph()

        # Reading Graph Info table
        if self.log:
            print( " \033[94mTrain {}:\033[0m Going over graph info".format(self.id) )

        with open(graphInfo) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    if not row[0] == "Number of vertices":
                        raise Exception("Wrong input file format. See map input format")
                    self.nVertices = int(row[1])
                elif line_count == 1:
                    if not row[0] == "Number of connections":
                        raise Exception("Wrong input file format. See map input format")
                    self.nEdges = int(row[1])
                line_count += 1

            if self.log:
                print( " \033[94mTrain {}:\033[0m  - Map contains {} vertices and {} edges".format(self.id, self.nVertices, self.nEdges) )

        # Reading Vertices Positions table
        if self.log:
            print( " \033[94mTrain {}:\033[0m Going over vertices positions".format(self.id) )

        self.vert_names = []
        self.vert_pos = []
        self.vert_idx = {}
        self.vert_namePos = {}

        # TODO: Check what dictionaries are useful to have as attributes , Map Variables

        with open(vertices) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = -1
            for row in csv_reader:
                if line_count == -1:
                    line_count += 1
                    continue
                self.vert_names += [ row[0] ]
                self.vert_pos += [ (float(row[1]), float(row[2])) ]
                self.vert_idx[ (float(row[1]), float(row[2])) ] = line_count
                self.vert_namePos[ row[0] ] = (float(row[1]), float(row[2]))
                self.graph.add_node( row[0] )
                line_count += 1
            if line_count != self.nVertices:
                raise Exception("Wrong input file format. The number of vertices given doesn't match the number of vertices specified")

            if self.log:
                print(" \033[94mTrain {}:\033[0m - Got positions of the {} vertices".format(self.id, self.nVertices))

        # Reading Connection Matrix table
        if self.log:
            print( " \033[94mTrain {}:\033[0m Going over graph edges".format(self.id) )

        self.edges = np.ndarray(shape=(self.nVertices,self.nVertices), dtype=float)
        self.edges.fill(-1)

        with open(connections) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            edge_count = 0
            for row in csv_reader:
                for i in range(self.nVertices):
                    if row[i] != "":
                        self.edges[line_count][i] = float(row[i])
                        if float(row[i]) > 0:
                            self.graph.add_edge( self.vert_names[line_count],
                                                 self.vert_names[i],
                                                 distance = float(row[i]) )
                        if line_count > i:
                            edge_count += 1
                line_count += 1
            if self.nEdges != edge_count:
                raise Exception("Wrong input file format. Number of edges given doesn't match the specified number")

            if self.log:
                print(" \033[94mTrain {}:\033[0m - Read over {} edges in graph".format(self.id, edge_count))


        # Route calculation helpers
        self.routes = {}
        self.route_lengh = {}

        for vert1 in self.vert_names:
            if vert1[0] != "_":
                for vert2 in self.vert_names:
                    if vert2[0] != "_":
                        self.route_lengh[(vert1, vert2)] = nx.dijkstra_path_length(self.graph, vert1, vert2, "distance")
                        vPath = nx.dijkstra_path(self.graph, vert1, vert2, "distance")

                        self.routes[(vert1, vert2)] = []
                        for vert in vPath:
                            self.routes[(vert1, vert2)] += [ self.vert_namePos[vert] ]
    # -----------------------------------------------------------------------------------------

    def calculate_route(self, init, fin): #, measure="distance"):
        """
            Calculates the route from 'init' to 'fin'.
        :param init: Initial point. Should be the position of a vertice on the map
        :param fin: Final point. Should be the position of a vertice on the map
        :param measure: Type of "shortness measurament". It is assumed to be the distance between
            pints, but could be the edge weight, for example
        :return: Returns first the path, followed by the total distance between two points
        """
        measure = "distance"

        len_temp = 0
        if not init in self.vert_pos:
            init_node, len_temp = self.discover_proximity_point(init)

        if type(init) == list:
            init = (init[0], init[1])
        if type(fin) == list:
            fin = (fin[0], fin[1])

        if init == fin:
            return [], 0

        init_node = self.vert_names[ self.vert_idx[ init ] ]
        fin_node = self.vert_names[ self.vert_idx[ fin ] ]


        if init_node[0] != "_" and fin_node[0] != "_":
            # Both are stopping points, so path is already recorded in memory
            path = self.routes[ (init_node, fin_node) ]
            distance = self.route_lengh[ (init_node, fin_node) ] + len_temp

            return path, distance

        # If at least one of the vertices is not a stopping point, the train will calculate
        distances_length = nx.dijkstra_path_length(self.graph, init_node, fin_node, measure)
        distances_path = nx.dijkstra_path(self.graph, init_node, fin_node, measure)

        path = []
        for vert in distances_path:
            path += [ self.vert_namePos[vert] ]

        if len_temp != 0:
            distances_length += len_temp

        return path, distances_length
    # -----------------------------------------------------------------------------------------

    def full_distance(self, start=0, stop=None):
        """
            Calculates the full distance to be traveled in path schedule.
            Can optionally have a start or stop node
        :param start: First position in the path one wants to start the
          calculation with. Must be an index!
        :param stop: Last position in the path one wants the calculation to
          end in (including itself). Must be an index!
        """
        totSum = 0

        if self.path != []:
            if self.currentEdge and start==0:
                totSum += distance.euclidean(self.pos, self.path[0])

            startIdx = start
            stopIdx = len(self.path)-1 if not stop else stop
            for index in range(startIdx, stopIdx):
                totSum += distance.euclidean(self.path[index],self.path[index+1])
                continue
        return totSum
    # -----------------------------------------------------------------------------------------

    def acknowlege_request(self):
        """
            Sends answer to client to let it know the request is being processed
        """
        msg = Message(msgType=MsgTypes.req_ack, sender=self.id, receiver=self.unprocessedReqs['ID'])
        self.network.broadcast(msg.encode(), self)
    # -----------------------------------------------------------------------------------------

    def start_election(self, distance):
        """
            Starts election by broadcasting elec message to other trains
        :param distance: distance from my position until the pickup location
        """
        temp_distance = distance
        msg_sent = Message(msgType = MsgTypes.elec, sender = self.id, distance = temp_distance , client = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent.encode(), self)
    # -----------------------------------------------------------------------------------------

    def silence_train(self, nodeId):
        """
            Sends acknowledgement message to train with higher distance to client, thus
            silencing it for election purposes
        :param nodeId: ID of the train that is the desired message receipient
        """
        temp_nodeID = nodeId
        msg_sent = Message(msgType = MsgTypes.elec_ack, sender = self.id, receiver = temp_nodeID , client = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent.encode(), self)
    # -----------------------------------------------------------------------------------------

    def client_accept(self):
        """
            Method to encapsule messages send when a client is accepted
            (train has won client election and will pick-up client)
        """
        if self.log:
            print(" \033[94mTrain {}:\033[0m Sending leader message to other trains and answering client request".format(self.id))

        msg_sent_trains = Message(msgType = MsgTypes.leader, sender = self.id, client = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent_trains.encode(), self)
        msg_sent_client = Message(msgType = MsgTypes.req_ans, sender = self.id, receiver = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent_client.encode(), self)
    # -----------------------------------------------------------------------------------------

    def notify_client(self, client):
        """
            Notifies client of train arrival at the pick up or drop off location
        """
        mType = None
        if client < 0:
            mType = MsgTypes.pickup
            if self.log:
                print(" \033[94mTrain {}:\033[0m Reached client. Sending message to notify him".format(self.id))

        elif client > 0:
            mType = MsgTypes.dropoff
            if self.log:
                print(" \033[94mTrain {}:\033[0m Reached destination. Sending message to notify client".format(self.id))

        msg = Message(msgType=mType, sender=self.id, receiver=abs(client))
        self.network.broadcast(msg.encode(), self)
    # -----------------------------------------------------------------------------------------

    def discover_proximity_point(self, point):
        dist = {}
        minVal = 10000000
        point_temp = point

        for vertice in self.vert_pos:
            value = distance.euclidean( vertice, point )
            dist[ vertice ] = value

            if (value < minVal):
                minVal = value
                point_temp = vertice

        return point_temp, minVal
    # -----------------------------------------------------------------------------------------

    def move(self):
        """
            Moves train position across the map
        """
        # NOTE: Train is currently traveling with constant speed throughout vertices
        # (No acceleration considered)

        # FIXME: Sometimes path acquires two consecutive vertices that are the same. This bugs the system
        #   Need to find out why this happens and how this happens to fix it

        if len(self.path) > 0 and self.okToMove:

            posInit = self.pos

            if len(self.path) >= 2:
                if self.path[0] == self.path[1]:
                    print("\033[91mERROR OCCURED!!!\033[0m Path has consecutive vertices with same value ({})".
                          format(self.path[:2]))

            # Fist: move train according to current speed
            pos0 = self.pos[0] + self.v[0] * self.vStep
            pos1 = self.pos[1] + self.v[1] * self.vStep
            self.pos = (pos0, pos1)

            distanceToVertice = (self.path[0][0] - self.pos[0], self.path[0][1] - self.pos[1])
            if (distanceToVertice[0] * self.v[0] < 0) or (distanceToVertice[1] * self.v[1] < 0):
                # Passed vertice! Roll back
                self.pos = (self.path[0][0], self.path[0][1])

            self.totalDistanceRun += distance.euclidean(posInit, self.pos)

            # Update path
            if (self.pos[0] == self.path[0][0]) and (self.pos[1] == self.path[0][1]):
                # Disocupy road
                if self.currentEdge:
                    self.semaphore[ self.currentEdge ] = True
                    self.currentEdge = None

                # Go to next speed step
                self.path = self.path[1:]
                self.currentGoal = self.goals.pop(0)
                self.v = [0, 0]

                if 'ID' in self.unprocessedReqs.keys():
                    # Possibly need to update indexes!
                    if 'pick_idx' in self.unprocessedReqs.keys():
                        self.unprocessedReqs['pick_idx'] -= 1
                        if 'drop_idx' in self.unprocessedReqs.keys():
                            self.unprocessedReqs['drop_idx'] -= 1

                # TODO: pick or drop client
                if self.currentGoal:
                    # Will pick up or drop off a client
                    if self.okToMove: # NOTE: Is this case needed? Should only reach here once, if okToMove
                        if self.log:
                            print(" \033[94mTrain {}:\033[0m Reached goal {}".format(self.id, self.pos))

                        self.okToMove = False
                        self.waitForClientDelay = 0
                    return

                if not self.path:
                    return

            if self.v == [0, 0]:
                v1 = self.vert_idx[ (self.pos[0], self.pos[1]) ]
                v2 = self.vert_idx[ (self.path[0][0], self.path[0][1]) ]

                a = max(v1, v2)
                b = min(v1, v2)

                if not self.semaphore[ (a, b) ]:
                    if (self.log):
                        print( " \033[94mTrain {}:\033[0m Road occupied. Try again later".format(self.id) )
                    return

                else:
                    # Occupying road
                    self.semaphore[(a, b)] = False
                    self.currentEdge = (a, b)

                    # Updating speed
                    nextEdge = (self.path[0][0] - self.pos[0], self.path[0][1] - self.pos[1])
                    magnitude = distance.euclidean(self.path[0], self.pos)
                    direction = (nextEdge[0] / magnitude, nextEdge[1] / magnitude)
                    self.v = [self.vMax * direction[0], self.vMax * direction[1]]
    # -----------------------------------------------------------------------------------------

    def draw(self, ax):
        """
            Draws the train on the map
        :param ax: Subplot object where train should be drawn
        :return:
        """
        rotation = np.angle(self.v[0] + self.v[1]*1j, deg=True)

        with cbook.get_sample_data(self.img) as image_file:
            image = plt.imread(image_file)

        if self.mode == TrainModes.busy:
            im = ax.imshow(image[:, :, 0], extent=[0, 1, 0, 1], clip_on=True)
        else:
            im = ax.imshow(image, extent=[0, 1, 0, 1], clip_on=True)

        xmin, xmax, ymin, ymax = ax.axis()
        scale = (ymax-ymin) * .05  # Scale fator to print visible trains

        trans_data = mtransforms.Affine2D().scale(scale, scale).translate(-scale * .5, 0).\
                         rotate_deg(rotation).translate(self.pos[0], self.pos[1]) + ax.transData
        im.set_transform(trans_data)

        direction = [1,1]
        magnitude = distance.euclidean((0,0), self.v)
        if magnitude != 0:
            seno = self.v[1]/magnitude
            cosseno = self.v[0]/magnitude
            direction = [ direction[0]*cosseno - direction[1]*seno,
                          direction[0]*seno + direction[1]*cosseno  ] # Rotating vector

        ax.text(self.pos[0] + .7 * scale * direction[0], self.pos[1] + .6 * scale * direction[1],
                "{}".format(self.id))

        if self.mode == TrainModes.busy:
            deviation = 0
            for client in self.inCourseClients:
                dirClient = [-1, 1 - deviation]
                if magnitude != 0:
                    dirClient = [ dirClient[0]*cosseno - dirClient[1]*seno,
                                  dirClient[0]*seno + dirClient[1]*cosseno  ]
                ax.text(self.pos[0] + .7 * scale * dirClient[0],
                        self.pos[1] + .6 * scale * dirClient[1],
                        "{}".format(int(client - .5)),
                        fontsize=8,
                        verticalalignment='bottom', horizontalalignment='center',
                        color='blue')
                deviation += .2

        x1, x2, y1, y2 = im.get_extent()
        ax.plot(x1, y1, transform=trans_data, zorder=10)
    # -----------------------------------------------------------------------------------------

    def kill(self):
        """
            Terminate this object. Should be called by simulation when taking train out of it
        """
        print( " \033[94mTrain {}:\033[0m Command for Killing Me".format(self.id) )
        del self