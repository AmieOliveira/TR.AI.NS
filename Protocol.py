"""
    Messaging protocol
    TR.AI.NS Project
    Author: Amanda
"""

__all__ = ['Message', 'MsgTypes']

import json
from enum import Enum


class MsgTypes(Enum):
    """
        Group of message types that can be exchanged in the system
    'req'      -> Service request from client
    'req_ack'  -> Message sent by the train to the client to let it know the request is
        being processed
    'req_ans'  -> Train accept of the client request (to be sent by the selected train
        amongst the ones that received the request)
    'elec'     -> Election message, used to select the train that will accept the client
        request
    'elec_ack' -> Acknowledgement of 'elec' message received. Sent by a train to inform
        the receiver it has received the message and has a smaller distance until the
        target, making it better suited to answer the call
    'leader'   -> Sent to inform other trains that I will be picking client up
    """
    req = 1
    req_ack = 6
    req_ans = 2
    elec = 3
    elec_ack = 4
    leader = 5
    pickup = 7
    dropoff = 8
    # canc -> give up on race
    # map_updt -> update map


class Message:

    def __init__(self, msgType = None, sender = None, **kwargs):
        """
            Message exchange protocol for the TR.AI.NS system
        :param msgType: Type of message to be sent ('REQ', 'REQ_ANS', '
        :param sender: ID of the message sender
        :param kwargs: Depending on the type of message, one must give different arguments
        """
        # TODO: Give full kwargs list on description
        if not msgType:
            return

        self.msgDict = {"type": msgType.value, "sender": sender}
        self.nType = msgType

        if self.nType == MsgTypes.req:               # Client is requesting pickup
            self.msgDict["pickUp"] = kwargs["pickup"]
            self.msgDict["dropOff"] = kwargs["dropoff"]

        elif self.nType == MsgTypes.req_ack:           # Train will process request
            self.msgDict["receiver"] = kwargs["receiver"]

        elif self.nType == MsgTypes.req_ans:           # Train has accepted client request
            self.msgDict["receiver"] = kwargs["receiver"]

        elif self.nType == MsgTypes.elec:              # Leader election message
            self.msgDict["distance"] = kwargs["distance"]
            self.msgDict["clientID"] = kwargs["client"]
            # self.msgDict["starter"] = kwargs["starter"]

        elif self.nType == MsgTypes.elec_ack:          # Acknowledge message sender.
                                            # (Sent if the sender has the shorter path.)
            self.msgDict["clientID"] = kwargs["client"]
            # self.msgDict["starter"] = kwargs["starter"]
            self.msgDict["receiver"] = kwargs["receiver"]

        elif self.nType == MsgTypes.leader:            # Leader chosen to answer request
            self.msgDict["clientID"] = kwargs["client"]

        elif self.nType == MsgTypes.pickup:             # Sent to let client know train has arrived
            self.msgDict["receiver"] = kwargs["receiver"]

        elif self.nType == MsgTypes.dropoff:             # Sent to let client know its destination has arrived
            self.msgDict["receiver"] = kwargs["receiver"]


    def __getitem__(self, key):
        """
            Overloads the [] operator, so that obj[k] == obj.msgDict[k]
        :param key: The name of the wanted attribute in the dict
        :return: The value of the key parameter in the class dict
        """
        return self.msgDict[key]

    def encode(self):
        """
            Encodes the message into a string to be transmitted
        :return: String version of the message
        """
        return json.dumps(self.msgDict)

    def decode(self, msgStr):
        """
            Gets the corresponding dictionary from the encoded string
        :param msgStr: String version of the message
        """
        self.msgDict = json.loads(msgStr)
        self.nType = MsgTypes(self.msgDict["type"])