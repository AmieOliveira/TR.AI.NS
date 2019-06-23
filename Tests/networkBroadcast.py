#!/usr/local/bin/python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from Train import Train
from Client import Client
from Network import Network
# import simulation...

from Protocol import Message, MsgTypes


class Simulation:
    def __init__(self):
        self.devices = []
        self.trainRange = 15
        self.clientRange = 5


tr1 = Train(1, (-1,0), "map.txt")
tr2 = Train(2, (2,3), "map.txt")
tr3 = Train(3, (0,7), "map.txt")
cl1 = Client(-1, (0,0), "map.txt")

sim = Simulation()
net = Network(sim, log=True)

sim.devices = [tr1, cl1, tr2, tr3]


m = Message(MsgTypes.req, sender=cl1.id, pickup='Point 1', dropoff='Point  2')
mE = Message(MsgTypes.elec, sender=tr1.id, distance=10, client=-1)


net.broadcast(m.encode(), cl1)

print ("Train 1: %s" % tr1.messageBuffer)
print ("Train 2: ", tr2.messageBuffer)
print ("Train 3: {}", tr3.messageBuffer)

net.broadcast(mE.encode(), tr1)

print ("Train 1: ", tr1.messageBuffer)
print ("Train 2: ", tr2.messageBuffer)
print ("Train 3: ", tr3.messageBuffer)

