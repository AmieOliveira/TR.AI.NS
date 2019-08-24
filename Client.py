"""
    Train Robot Classes
    TR.AI.NS Project
    Authors: Amanda, Vinicius
"""

__all__ = ['Client']

from Protocol import Message, MsgTypes
from enum import Enum
import os
from random import random
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.transforms as mtransforms


class CliModes(Enum):
    """
        Group of possible client operational states at any given moment
    'login'   -> Has entered the system, but hasn't made a request yet
    'request' -> Client has made a request and is waiting for its answer
    'wait'    -> Request has been accepted and train is on its way
    'moving'  -> In car, going to drop off point
    'dropoff' -> Client has been delivered
    """
    login = 0
    request = 1
    wait = 2
    moving = 3
    dropoff = 4


class Client:
    def __init__(self, ID, pos0, destiny, mapFile, network, log=False):
        """
            Class Client contains the whole operational code for a client in
            the TR.AI.NS project system
        :param ID: Gives the ID number of the client. Each client should have a unic
            identifier, which should also be different from all trains identifiers
            Norm says clients should have integers plus a half as identifiers (for
             example 0.5 or 13.5)
        :param pos0: Initial position of client. Format should be '(x, y)'. Needs to
          be one of the stopping points on the map
        :param destiny: Desired destination. Needs to be one of the stopping points
          on the map
        :param mapFile: Name of the file that contains the map information
        :param network: Object that simulates a broadcast medium of communication
        :param log: Boolean parameter that enables or disables client's prints
        """
        self.id = ID
        if int(self.id) == self.id:
            self.id = ID + .5
            print("Client IDs should be some number and a half. Client ID is {}".format(self.id))

        # Logging variable
        self.log = log

        # Network object
        self.network = network # Connecto to the network communication system

        # Moving attributes
        self.pos = tuple(pos0) # Current position of the train
        self.destiny = destiny # Current client destiny

        # Message buffer
        self.messageBuffer = []

        # Client gif image
        self.img = os.path.dirname(os.path.abspath(__file__)) + '/man-user.png'
        self.rand_pos = 2*random()

        # Initial client mode
        self.mode = CliModes.login

        # Request status variables
        self.leaveLogin = True
        self.reqAnswer = False
        self.answerTimer = 0
        self.answerTimeout = 10

        # Train that accepted me
        self.train = None

        # Waiting time statistics
        self.timeTillRequest = -1       # Time until client receives a train acceptance
        self.waitingTime = -1           # Time until client is picked up by train

        if self.log:
            print("  \033[92mClient {}:\033[0m Created client in {}, with destination {}".format(self.id, self.pos, self.destiny))
    # ---------------------------------------------------

    def step(self):
        """
            This method executes the whole operation of the client during a logic step.
            Should be looped to have it functioning.
        """

        # Updating timers
        if self.mode == CliModes.request:
            self.answerTimer += 1
            self.timeTillRequest +=1
            self.waitingTime += 1
        elif self.mode == CliModes.wait:
            self.waitingTime += 1

        # Receiving and interpreting messages
        currentMessage = None
        if len(self.messageBuffer) > 0:
            # In this case there are messages to be interpreted
            currentMessage = self.messageBuffer.pop(0)

        if currentMessage:

            # NOTE: Already checked that messages are for me in receive_message

            # Case 1: Request acknowledge
            if currentMessage['type'] == MsgTypes.req_ack.value:
                self.reqAnswer = True
                if self.log:
                    print("  \033[92mClient {}:\033[0m There are trains processing my request".format(self.id))
                # There is at least one train that will process my request

            # Case 2: Request accept
            elif currentMessage['type'] == MsgTypes.req_ans.value:
                self.mode = CliModes.wait
                self.train = currentMessage['sender']
                if self.log:
                    print("  \033[92mClient {}:\033[0m Will be picked up by train {} (waited {} simulation steps)".
                          format(self.id, self.train, self.timeTillRequest))

            # Case 3: Train arrival
            elif currentMessage['type'] == MsgTypes.pickup.value:
                self.mode = CliModes.moving
                if self.log:
                    print("  \033[92mClient {}:\033[0m Boarding train. (waited {} simulation steps since request)".
                          format(self.id, self.waitingTime))

            # Case 4: Destination arrival
            elif currentMessage['type'] == MsgTypes.dropoff.value:
                self.mode = CliModes.dropoff
                self.pos = (self.destiny[0], self.destiny[1])
                if self.log:
                    print("  \033[92mClient {}:\033[0m Getting off train".format(self.id))
        # -----------------------------------

        # Updating client mode of operation
        if self.mode == CliModes.login:
            if self.leaveLogin:
                if self.log:
                    print("  \033[92mClient {}:\033[0m Sending request.".format(self.id))
                self.request_ride()
                self.answerTimer = 0
                self.reqAnswer = False
                self.mode = CliModes.request
                self.timeTillRequest = 0
                self.waitingTime = 0

        elif self.mode == CliModes.request:
            if not self.reqAnswer:
                if self.answerTimer >= self.answerTimeout:
                    if self.log:
                        print( "  \033[92mClient {}:\033[0m Timeout. Resending request.".format(self.id) )

                    self.request_ride()
                    self.answerTimer = 0
                    self.reqAnswer = False
    # ---------------------------------------------------

    def receive_message(self, msgStr):
        """
            Receives message in string format and converts it into a protocol class
        :param msgStr: Should be a string coded with the message
        """
        msg = Message()
        msg.decode(msgStr)

        if msg['sender'] == self.id:
            return

        if msg.nType in [MsgTypes.req_ans, MsgTypes.req_ack, MsgTypes.pickup, MsgTypes.dropoff]:
            if msg['receiver'] == self.id:
                self.messageBuffer += [msg]
    # ---------------------------------------------------

    def request_ride(self):
        """
            Send request message to the trains
        """
        msg_sent = Message(msgType = MsgTypes.req, sender=self.id, pickup = self.pos, dropoff=self.destiny)
        self.network.broadcast(msg_sent.encode(), self)
    # ---------------------------------------------------

    def draw(self, ax):
        """
            Draws the client on the map
        :param ax: Subplot object where train should be drawn
        :return:
        """
        if self.mode != CliModes.moving:
            with cbook.get_sample_data(self.img) as image_file:
                image = plt.imread(image_file)

            im = ax.imshow(image, extent=[0, 1, 0, 1], clip_on=False, zorder=7)

            multiplier = 1
            if self.mode == CliModes.dropoff:
                multiplier = -1

            xmin, xmax, ymin, ymax = ax.axis()
            scale = (ymax - ymin) * .03  # Scale fator to print visible trains

            trans_data = mtransforms.Affine2D().scale(-scale * multiplier, scale).\
                                         translate(self.pos[0] - (.5 * scale + self.rand_pos) * multiplier, self.pos[1] + .5 * scale)\
                                         + ax.transData
            im.set_transform(trans_data)

            ax.text(self.pos[0] - (2 * scale + self.rand_pos) * multiplier, self.pos[1] + .5 * scale,
                    "{}".format(int(self.id-.5)), verticalalignment='bottom', horizontalalignment='center',
                    color='blue')

            x1, x2, y1, y2 = im.get_extent()
            ax.plot(x1, y1, transform=trans_data, zorder=7)
    # ---------------------------------------------------

    def kill(self):
        """
             Terminate this object. Should be called by simulation when taking client out of it
        """
        print( "  \033[92mClient {}:\033[0m Command for Killing Me".format(self.id) )
        del self