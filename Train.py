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
    def __init__(self, ID, pos0, mapFile, network, log=False):
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

        self.vStep = .1                  # approximate s/step ratio
        self.v = [0, 0]                      # train speed in m/s
        # Quando evoluir adiconar aceleração

        self.vMax = 6                   # Maximum train speed in m/s
        #self.aMax = 1                   # Maximum train acceleration in m/s^2
        # TODO: Check this variables

        # Messaging attributes
        self.messageBuffer = []

        # Map attributes
        self.load_map(mapFile)

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
        self.maximumMsgWait = 100

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

        # Reading and interpreting messages in the message buffer
        currentMessage = None
        if len(self.messageBuffer) > 0:
            # In this case there are messages to be interpreted
            currentMessage = self.messageBuffer.pop()

        if currentMessage:
            if self.log:
                print("Received message: {}".format(currentMessage.nType.name))
                # print "\t %s" % str(currentMessage.msgDict)

            # Case 1: Service request from client
            if currentMessage['type'] == MsgTypes.req.value:

                if self.trainMode != TrainModes.outOfOrder: # Checks if train can accept
                    if not ('ID' in self.unprocessedReqs.keys()): # Checks if there are current processes ongoing

                        if self.log:
                            print("Processing Client Request")

                        clientID = currentMessage['sender']
                        route, d = None, None
                        # Calculate route
                        if self.trainMode == TrainModes.wait:
                            route, d = self.calculate_route( self.pos, currentMessage['pickUp'] )
                        elif (self.trainMode == TrainModes.accept) or (self.trainMode == TrainModes.busy):
                            route, d = self.calculate_route( self.path[-1], currentMessage['pickUp'] )
                        self.unprocessedReqs = dict(ID=clientID, pickup=currentMessage['pickUp'],
                                                    dropoff=currentMessage['dropOff'], delayT=0,
                                                    inElections=False, simpleD=d, route=route, msgWait=0)
                        # TODO: Train answers Request!
                        # Create a message type to indicate to client that the request has been heard and is being processed

            # Case 2: Election started
            elif currentMessage['type'] == MsgTypes.elec.value:

                if not self.trainMode == TrainModes.outOfOrder:  # Checks if train can accept
                    # if not self.outOfElec == currentMessage['clientID']: # Check if has already 'lost' election

                    if 'ID' in self.unprocessedReqs.keys():
                        if self.unprocessedReqs['ID'] == currentMessage['clientID']:
                            # NOTE: I assume any car receives first the notice from the client
                            if self.log:
                                print("Received Election Message")

                            dTot = self.unprocessedReqs['simpleD'] + self.full_distance()

                            if dTot < currentMessage['distance']:
                                # This train is the leader himself
                                self.silence_train(currentMessage['sender'])
                                if not self.unprocessedReqs['inElections']:
                                    # If It hasn't yet send its distance, should do so now
                                    self.start_election(dTot)
                                    self.unprocessedReqs['inElections'] = True
                                    self.unprocessedReqs['msgWait'] = 0

                                if self.log:
                                    print("\t Win this elections round")

                            # TODO: Solve the case where they both have the same distance
                            # Desmpate pelo id???
                            else:
                                # Finishes current election process
                                self.outOfElec = self.unprocessedReqs['ID']
                                self.unprocessedReqs = {}

                                if self.log:
                                    print("\t Lost these elections")

            # Case 3: Election answer
            elif currentMessage['type'] == MsgTypes.elec_ack.value:
                if "ID" in self.unprocessedReqs.keys():
                    if self.unprocessedReqs['ID'] == currentMessage['clientID']: # Checks if this message is from current election
                        # No need to check if message is destined to itself, because the receiving mechanism already does so.
                        # Train lost current election. Finishes election process
                        self.outOfElec = self.unprocessedReqs['ID']
                        self.unprocessedReqs = {}

                        if self.log:
                            print("Silenced in these elections. Lost election.")

            # Case 4: Leader Message
            elif currentMessage['type'] == MsgTypes.leader:
                if "ID" in self.unprocessedReqs.keys():
                    if self.unprocessedReqs['ID'] == currentMessage['clientID']: # Checks if this message is from current election
                        self.outOfElec = self.unprocessedReqs['ID']
                        self.unprocessedReqs = {}

                        if self.log:
                            print("Got an election leader in these elections. Lost election.")

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
                        print("Starting Election!")

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
                        print("Finishing election! I've won! (ID {})".format(self.id))

                    self.path += self.unprocessedReqs['route'] # Adds route to desired path
                    # TODO: Think on pickup and dropoff. Might be strings instead of actual coordinates...
                    # In this case I'd need to convert into coordinates
                    self.client += [(self.unprocessedReqs['ID'], self.unprocessedReqs['pickup'], self.unprocessedReqs['dropoff'])]
                    self.client_accept()
                    self.unprocessedReqs = {} # Finishes current election process

                    if self.trainMode == TrainModes.wait:
                        self.trainMode = TrainModes.accept
                        self.currentGoal = self.client[0][1] # pickup
        # ------------------------------------------

        # Moving train and handling new position
        self.move()

        if self.pos == self.currentGoal:  # Reached current destination
            if self.trainMode == TrainModes.accept:
                # Client boarding train
                self.trainMode = TrainModes.busy
                self.currentGoal = self.client[0][2] # dropoff

            elif self.trainMode == TrainModes.busy:
                # Client leaving the train
                self.client.pop() # taking out client from list
                if len(self.client) > 0:
                    self.trainMode = TrainModes.accept
                    self.currentGoal = self.client[0][1] # pickUp
                else:
                    self.currentGoal = None
                    self.trainMode = TrainModes.wait

            elif self.trainMode == TrainModes.outOfOrder:
                self.kill()
    # -----------------------------------------------------------------------------------------

    def receive_message(self, msgStr):
        """
            Receives message in string format and converts it into a protocol class
        :param msgStr: Should be a string coded with the message
        """
        msg = Message()
        msg.decode(msgStr)

        if msg.nType == MsgTypes.req:
            self.messageBuffer += [msg]

        elif msg.nType == MsgTypes.elec:
            # if 'ID' in self.unprocessedReqs.keys():
            #     if msg['clientID'] == self.unprocessedReqs['ID']:
            self.messageBuffer += [msg]

        else:
            if msg['receiver'] == self.id:
                self.messageBuffer += [msg]
    # -----------------------------------------------------------------------------------------

    def send_message(self, msgType, **kwargs):
        """
            Send messaages to others trains
        :param msgType:
        :param kwargs:
        :return:
        """
        # TODO: Check if this method is needed
        temp_dict = kwargs
        msg_sent = Message(msgType = msgType, sender = self.id, kwargs=temp_dict)
        self.network.broadcast(msg_sent.encode(), self)
    # -----------------------------------------------------------------------------------------

    def load_map(self, mapPath):
        """
            Loads map information into the train object. Sets up necessary attributes
        :param mapPath: The folder path for the CSV files with the map content. Files
            must be created according to the model file format
        """

        if self.log:
            print("Reading map file ({})".format(mapPath))

        # Getting CSV file names
        graphInfo = "%s/Sheet 1-Graph Info.csv" % mapPath
        vertices = "%s/Sheet 1-Vertices Positions.csv" % mapPath
        connections = "%s/Sheet 1-Connection Matrix.csv" % mapPath

        # Reading Graph Info table
        if self.log:
            print("\tGoing over graph info")

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
                print("\t - Map contains {} vertices and {} edges".format(self.nVertices, self.nEdges))

        # Reading Vertices Positions table
        if self.log:
            print("\tGoing over vertices positions")

        self.vert_names = []
        self.vert_pos = []
        self.stoppingPoints = {}
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
                if row[0][0] != "_":
                    self.stoppingPoints[row[0]] = line_count
                line_count += 1
            if line_count != self.nVertices:
                raise Exception("Wrong input file format. The number of vertices given doesn't match the number of vertices specified")

            if self.log:
                print("\t - Got positions of the {} vertices. {} are stopping points".format(self.nVertices, len(self.stoppingPoints.keys())))

        # Reading Connection Matrix table
        if self.log:
            print("\tGoing over graph edges")

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
                        if line_count > i:
                            edge_count += 1
                line_count += 1
            if self.nEdges != edge_count:
                raise Exception("Wrong input file format. Number of edges given doesn't match the specified number")

            if self.log:
                print("\t - Read over {} edges in graph".format(edge_count))
    # -----------------------------------------------------------------------------------------

    def calculate_route(self, init, fin):
        # TODO
        return [], 4
    # -----------------------------------------------------------------------------------------

    def full_distance(self):
        sum = 0
        if self.path != []:
            for index in range(len(self.path)-1):
                sum += distance.euclidean(self.path[index],self.path[index+1])
                continue
        return sum
    # -----------------------------------------------------------------------------------------

    def start_election(self, distance):
        """
            Starts election by broadcasting elec message to other trains
        :param distance: distance from my position until the pickup location
        :return:
        """
        temp_distance = distance
        msg_sent = Message(msgType = MsgTypes.elec, sender = self.id, distance = temp_distance , client = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent.encode(), self)
    # -----------------------------------------------------------------------------------------

    def silence_train(self, nodeId):
        temp_nodeID = nodeId
        msg_sent = Message(msgType = MsgTypes.elec_ack, sender = self.id, receiver = temp_nodeID , client = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent.encode(), self)
        # TODO: Check if send_message is needed
    # -----------------------------------------------------------------------------------------

    def client_accept(self): # Envia mensagem de líder para todos os trens e request answer para o cliente.
        msg_sent_trains = Message(msgType = MsgTypes.leader, sender = self.id, client = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent_trains.encode(), self)
        msg_sent_client = Message(msgType = MsgTypes.req_ans, sender = self.id, receiver = self.unprocessedReqs['ID'])
        self.network.broadcast(msg_sent_client.encode(), self)
    # -----------------------------------------------------------------------------------------

    def move(self):
        # NOTE: Train is currently traveling with constant speed throughout vertices
        # (No acceleration considered)

        if len(self.path) > 0:
            # Fist: move train according to current speed
            self.pos[0] += self.v[0] * self.vStep
            self.pos[1] += self.v[1] * self.vStep

            distanceToVertice = (self.path[0][0] - self.pos[0], self.path[0][1] - self.pos[1])
            if (distanceToVertice[0] * self.v[0] < 0) or (distanceToVertice[1] * self.v[1] < 0):
                # Passed vertice! Roll back
                self.pos = [self.path[0][0], self.path[0][1]]

            # Second: update path
            if (self.pos[0] == self.path[0][0]) and (self.pos[1] == self.path[0][1]):
                self.path = self.path[1:]
                self.v = [0, 0]

                if self.pos == self.currentGoal:
                    # Will pick up or drop off a client
                    return

            if self.v == [0, 0]:
                # TODO: Put precaution to verify if path is available
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

        im = ax.imshow(image, extent=[0, 1, 0, 1], clip_on=True)

        if (self.trainMode == TrainModes.busy):
            im.set_cmap('nipy_spectral')

        trans_data = mtransforms.Affine2D().scale(2, 2).rotate_deg(rotation).translate(self.pos[0], self.pos[1]) + ax.transData
        im.set_transform(trans_data)
        x1, x2, y1, y2 = im.get_extent()
        ax.plot(x1, y1, transform=trans_data, zorder=5)
    # -----------------------------------------------------------------------------------------

    def kill(self):
        print("Command for Killing Me")
        del self