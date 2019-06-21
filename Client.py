"""
    Train Robot Classes
    TR.AI.NS Project
    Author: Amanda
"""

__all__ = ['Client']

from Protocol import Message, MsgTypes
from enum import Enum


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

    def draw(self):
        # TODO
        pass

    def kill(self):
        # TODO
        pass