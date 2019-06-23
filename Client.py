"""
    Train Robot Classes
    TR.AI.NS Project
    Authors: Amanda, Vinicius
"""

__all__ = ['Client']

from Protocol import Message, MsgTypes
from enum import Enum
import os
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
        self.id = ID
        if int(self.id) == self.id:
            self.id = ID + .5
            print("Client IDs should be some number and a half. Client ID is {}".format(self.id))

        # Logging variable
        self.log = log

        # Network object
        self.network = network # Connecto to the network communication system

        # Moving attributes
        self.pos = pos0  # Current position of the train
        self.destiny = destiny # Current client destiny

        # Message buffer
        self.messageBuffer = []

        # Client gif image
        self.img = os.path.dirname(os.path.abspath(__file__)) + '/man-user.png'

        # Initial client mode
        self.mode = CliModes.login

        # TODO: Check

    # ---------------------------------------------------
    def step(self):
        # TODO
        pass

    # ---------------------------------------------------
    def receive_message(self, msgStr):
        """
        Receives message in string format and converts it into a protocol class
        :param msgStr: Should be a string coded with the message
        """
        msg = Message()
        msg.decode(msgStr)

        if msg.nType == MsgTypes.req_ans:
            self.messageBuffer += [msg]
        else:
            if msg['receiver'] == self.id:
                self.messageBuffer += [msg]

    # ---------------------------------------------------
    def send_message(self, msg):
        # TODO
        pass

    # ---------------------------------------------------
    def request_ride(self):
        """
        Send request message to the trains
        """
        msg_sent = Message(msgType = MsgTypes.req, pickup = self.pos, dropoff=self.destiny)
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

            im = ax.imshow(image, extent=[0, 1, 0, 1], clip_on=True)

            trans_data = mtransforms.Affine2D().scale(-2, 2).translate(self.pos[0], self.pos[1]) + ax.transData
            im.set_transform(trans_data)
            x1, x2, y1, y2 = im.get_extent()
            ax.plot(x1, y1, transform=trans_data, zorder=7)



    def kill(self):
        print("Command for Killing Me")
        del self