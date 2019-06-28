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
    def __init__(self, ID, pos0, mapFile, edgeAvaliability, network, log=False):
        """
            Class Train contains the whole operational code for a transportation unit in
            the TR.AI.NS project.
        :param ID: Gives the ID number of the train. Every train should have a distinct ID
            (which should also be different from the client IDs)
        :param pos0: Initial position of the train. Format should be '(x, y)'
        :param mapFile: name of the file that contains the map information
        """
        self.id = ID

        # Network object
        self.network = network

        # Logging variable
        self.log = log

        # Moving attributes
        self.pos = pos0                 # Current position of the train

        self.currentEdge = None

        self.vStep = .2                  # approximate s/step ratio
        self.v = [0, 0]                      # train speed in m/s
            # Quando evoluir adiconar aceleração

        self.vMax = 6                   # Maximum train speed in m/s
        #self.aMax = 1                   # Maximum train acceleration in m/s^2

        self.okToMove = True
        self.waitForClientDelay = 0
        self.clientWaitingTime = 5

        # Messaging attributes
        self.messageBuffer = []

        # Map attributes
        self.load_map(mapFile)
        self.semaphore = edgeAvaliability

        # Operational attributes    (related to the paths being and to be taken)
        self.trainMode = TrainModes.wait        # Current mode of operation of the train

        self.currentGoal = None
        self.client = []                # List of pickup and dropOff locations for the next clients, with the client ID
                                        # [(Id1, pickup1, dropoff1), ...]

        self.path = []                 # List of vertices '(x, y)' to be passed through

        # Elections variables
        self.unprocessedReqs = {}       # Client request that is on process of train elections
                                        # Requests handled in dictionaries. ONLY ONE ALLOWED PER TURN
        self.outOfElec = None           # There has been a client request and this is not the elected train
        self.delayWanted = randint(1,11)
        self.maximumMsgWait = 15

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
        if (self.trainMode == TrainModes.accept) or (self.trainMode == TrainModes.busy):
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

                if self.trainMode != TrainModes.outOfOrder: # Checks if train can accept
                    if not ('ID' in self.unprocessedReqs.keys()): # Checks if there are current processes ongoing
                        clientID = currentMessage['sender']

                        if self.log:
                            print(" \033[94mTrain {}:\033[0m Processing Client {} Request".format(self.id, clientID))

                        route, d = None, None
                        # Calculate route
                        if self.trainMode == TrainModes.wait:
                            # In this case I am not moving , so I am in thory waiting at a vertice
                            route, d = self.calculate_route( self.pos, currentMessage['pickUp'] )
                        elif (self.trainMode == TrainModes.accept) or (self.trainMode == TrainModes.busy):
                            # FIXME: IndexError: list index out of range
                            # How can path be empty if train is still in those modes? There should be a '-1' index always
                            if len(self.path) == 0:
                                print("\033[91mERROR! PATH SHOULDN'T BE NULL!!\033[0m")
                                print(f"Train: {self.id}. Mode: {self.trainMode}")
                                print(self.path, self.client)
                                print(self.unprocessedReqs)
                            route, d = self.calculate_route( self.path[-1], currentMessage['pickUp'] )
                            #route = route[1:]

                        self.unprocessedReqs = dict(ID=clientID, pickup=currentMessage['pickUp'],
                                                    dropoff=currentMessage['dropOff'], delayT=0,
                                                    inElections=False, simpleD=d, route=route, msgWait=0)

                        self.acknowlege_request()
                        # Create a message type to indicate to client that the request has been heard and is being processed

            # Case 2: Election started
            elif currentMessage['type'] == MsgTypes.elec.value:

                if not self.trainMode == TrainModes.outOfOrder:  # Checks if train can accept
                    # if not self.outOfElec == currentMessage['clientID']: # Check if has already 'lost' election

                    if 'ID' in self.unprocessedReqs.keys():
                        if self.unprocessedReqs['ID'] == currentMessage['clientID']:
                            # NOTE: I assume any car receives first the notice from the client
                            if self.log:
                                print(" \033[94mTrain {}:\033[0m Received Election Message (from {})".format(self.id, currentMessage['sender']))

                            dTot = self.unprocessedReqs['simpleD'] + self.full_distance()

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
                if self.unprocessedReqs['delayT'] == self.delayWanted:
                    # Will start election
                    if self.log:
                        print( " \033[94mTrain {}:\033[0m Starting Election!".format(self.id) )

                    self.unprocessedReqs['inElections'] = True
                    d = self.unprocessedReqs['simpleD'] + self.full_distance() # Needs to add the distance until the
                                        # final position in path
                    self.start_election(d)
                    self.unprocessedReqs['msgWait'] = 0
        # ------------------------------------------

        # Elections finish
            else:
                if self.unprocessedReqs['msgWait'] == self.maximumMsgWait:
                    # If no answer is given, election isn't silenced and I am current leader
                    # self.broadcast_leader(self.id) # Inform others who's answering the request

                    if self.log:
                        print( " \033[94mTrain {}:\033[0m Finishing election! I've won!".format(self.id) )

                    self.path += self.unprocessedReqs['route'] # Adds route to desired path
                    if self.unprocessedReqs['simpleD'] == 0 and self.trainMode == TrainModes.wait:
                        self.okToMove = False
                        self.waitForClientDelay = 0

                    route, d = self.calculate_route(self.unprocessedReqs['pickup'], self.unprocessedReqs['dropoff'])

                    self.path += route[1:]

                    # In this case I'd need to convert into coordinates
                    self.client += [(self.unprocessedReqs['ID'], self.unprocessedReqs['pickup'], self.unprocessedReqs['dropoff'])]
                    self.client_accept()
                    self.unprocessedReqs = {} # Finishes current election process

                    if self.trainMode == TrainModes.wait:
                        self.trainMode = TrainModes.accept
                        self.currentGoal = tuple(self.client[0][1]) # pickup
        # ------------------------------------------

        # Moving train and handling new position
        if (self.trainMode == TrainModes.accept) and (not self.okToMove):
            if self.waitForClientDelay >= self.clientWaitingTime:
                self.okToMove = True
        if (self.trainMode == TrainModes.busy) and (not self.okToMove):
            if self.waitForClientDelay >= self.clientWaitingTime:
                self.okToMove = True

        self.move()

        if self.pos == self.currentGoal:  # Reached current destination
            if self.trainMode == TrainModes.accept:
                self.notify_client()

                # Client boarding train
                self.trainMode = TrainModes.busy
                self.currentGoal = tuple(self.client[0][2]) # dropoff

            elif self.trainMode == TrainModes.busy:
                # Client leaving the train
                self.notify_client()

                self.client.pop() # taking out client from list
                if len(self.client) > 0:
                    self.trainMode = TrainModes.accept
                    self.currentGoal = self.client[0][1] # pickUp
                else:
                    self.currentGoal = None
                    self.trainMode = TrainModes.wait

            elif self.trainMode == TrainModes.outOfOrder:
                self.kill()         # TODO: Check this. Don't know if this is the best usage
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
                        raise ("Wrong input file format. See map input format")
                    self.nVertices = int(row[1])
                else:
                    if not row[0] == "Number of connections":
                        raise ("Wrong input file format. See map input format")
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

                # node_positions = {node[0]: ( self.vert_namePos[node[0]] ) for node in self.graph.nodes(data=True)}
                # plt.figure(10)
                # nx.draw(self.graph, pos=node_positions, node_size=10, node_color='black', with_labels=True)
                # plt.title('Graph Representation of Train Map', size=15)
                # plt.show()
    # -----------------------------------------------------------------------------------------

    def calculate_route(self, init, fin, measure="distance"):
        """
            Calculates the route from 'init' to 'fin'.
        :param init: Initial point. Should be the position of a vertice on the map
        :param fin: Final point. Should be the position of a vertice on the map
        :param measure: Type of "shortness measurament". It is assumed to be the distance between
            pints, but could be the edge weight, for example
        :return: Returns first the path, followed by the total distance between two points
        """

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

        distances_length = nx.dijkstra_path_length(self.graph, init_node, fin_node, measure)
        distances_path = nx.dijkstra_path(self.graph, init_node, fin_node, measure)

        path = []
        for vert in distances_path:
            path += [ self.vert_namePos[vert] ]

        if len_temp != 0:
            distances_length += len_temp

        return path, distances_length
    # -----------------------------------------------------------------------------------------

    def full_distance(self):
        """
            Calculates the full distance to be traveled in path schedule
        """
        totSum = 0
        if self.path != []:
            for index in range(len(self.path)-1):
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

    def client_accept(self): # Envia mensagem de líder para todos os trens e request answer para o cliente.
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

    def notify_client(self):
        """
            Notifies client of train arrival at the pick up or drop off location
        """
        mType = None
        if self.trainMode == TrainModes.accept:
            mType = MsgTypes.pickup
            if self.log:
                print(" \033[94mTrain {}:\033[0m Reached client. Sending message to notify him".format(self.id))

        elif self.trainMode == TrainModes.busy:
            mType = MsgTypes.dropoff
            if self.log:
                print(" \033[94mTrain {}:\033[0m Reached destination. Sending message to notify client".format(self.id))

        msg = Message(msgType=mType, sender=self.id, receiver=self.client[0][0])
        self.network.broadcast(msg.encode(), self)
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

            if len(self.path) >= 2:
                if self.path[0] == self.path[1]:
                    print(f"\033[91mERROR OCCURED!!!\033[0m Path has consecutive vertices with same value ({self.path[:2]})")

            # Fist: move train according to current speed
            pos0 = self.pos[0] + self.v[0] * self.vStep
            pos1 = self.pos[1] + self.v[1] * self.vStep
            self.pos = (pos0, pos1)

            distanceToVertice = (self.path[0][0] - self.pos[0], self.path[0][1] - self.pos[1])
            if (distanceToVertice[0] * self.v[0] < 0) or (distanceToVertice[1] * self.v[1] < 0):
                # Passed vertice! Roll back
                self.pos = (self.path[0][0], self.path[0][1])

            # Update path
            if (self.pos[0] == self.path[0][0]) and (self.pos[1] == self.path[0][1]):
                # Disocupy road
                if self.currentEdge:
                    self.semaphore[ self.currentEdge ] = True
                    self.currentEdge = None

                # Go to next speed step
                self.path = self.path[1:]
                self.v = [0, 0]

                if self.pos == self.currentGoal:
                    # Will pick up or drop off a client
                    if self.okToMove:
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

                # FIXME: When there are two sequentoal vertices in path there are equal we get a bug here. But this was not supposed to happen
                if not self.semaphore[ (a, b) ]:
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

        if self.trainMode == TrainModes.busy:
            im = ax.imshow(image[:, :, 0], extent=[0, 1, 0, 1], clip_on=True)
        else:
            im = ax.imshow(image, extent=[0, 1, 0, 1], clip_on=True)

        trans_data = mtransforms.Affine2D().scale(2, 2).translate(-1, 0).rotate_deg(rotation).translate(self.pos[0], self.pos[1]) + ax.transData
        im.set_transform(trans_data)

        x1, x2, y1, y2 = im.get_extent()
        ax.plot(x1, y1, transform=trans_data, zorder=10)
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

    def kill(self):
        print( " \033[94mTrain {}:\033[0m Command for Killing Me".format(self.id) )
        del self