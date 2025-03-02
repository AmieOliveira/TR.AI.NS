"""
    Communication Network Class
    TR.AI.NS Project
    Author: Amanda, Vinicius
"""

__all__ = ['Network']

from Train import Train
from Client import Client
from math import sqrt


class Network:
    def __init__(self, simulator, log=False):
        """
            The network class is responsible for masking the communication network
            expected behavior in the TR.AI.NS project simulation.
        """
        self.sim = simulator
        self.log = log

    def broadcast(self, msgStr, sender):
        """
            This funcrion is responsible for delivering the desired message to its
            receipients
        :param msgStr: the message that is to be sent
        """
        xs = sender.pos[0]
        ys = sender.pos[1]

        if self.log:
            print("\033[93mNetwork:\033[0m Sender {}, from position ({},{})".format(sender.id, xs, ys))

        d = 0

        if isinstance(sender, Train):
            d = self.sim.trainRange
            if self.log:
                print("\033[93mNetwork:\033[0m Sender is a train. Reachable distance is {} m".format(d))
        elif isinstance(sender, Client):
            d = self.sim.clientRange
            if self.log:
                print("\033[93mNetwork:\033[0m Sender is a client. Reachable distance is {} m".format(d))

        for device in self.sim.devices:
            if sqrt( (xs - device.pos[0])**2 + (ys - device.pos[1])**2 ) <= d:
                device.receive_message(msgStr)
                if self.log:
                    print("\033[93mNetwork:\033[0m Sent message to device {}".format(device.id))