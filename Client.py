"""
    Train Robot Classes
    TR.AI.NS Project
    Author: Amanda
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
    def __init__(self, ID, pos0, mapFile, log=False):
        self.id = ID

        # Logging variable
        self.log = log

        # Moving attributes
        self.pos = pos0  # Current position of the train

        self.mode = CliModes.login
        self.img = os.getcwd() + '/man-user.png'
        # TODO: All
        pass

    def step(self):
        # TODO
        pass

    def receive_message(self, msgStr):
        # TODO
        pass

    def send_message(self, msg):
        # TODO
        pass

    def request_ride(self):
        # TODO
        pass

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
            ax.plot(x1, y1, transform=trans_data)



    def kill(self):
        # TODO
        pass