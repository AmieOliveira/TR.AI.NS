#!/usr/local/bin/python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from Train import Train
from Protocol import Message, MsgTypes


pos0 = (0, 0)
tr = Train(1, pos0, "none.txt", log=True)

msg = Message(MsgTypes.req, sender=-1, pickup='Point 1', dropoff='Point  2')
msgStr = msg.encode()
msg2 = Message(MsgTypes.elec, sender=3, distance=10, client=-1)
msg2Str = msg2.encode()
msg3 = Message(MsgTypes.elec_ack, sender=2, client=-1, receiver=3)
msg3Str = msg3.encode()
msg4 = Message(MsgTypes.elec_ack, sender=2, client=-1, receiver=1)
msg4Str = msg4.encode()

# print "Initial buffer: %s" % tr.messageBuffer
tr.receive_message(msgStr)
# print "After receiveing first message: %s" % tr.messageBuffer
tr.step()
tr.step()
tr.receive_message(msg2Str)
tr.step()
tr.receive_message(msg3Str)
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
tr.step()
# tr.receive_message(msg4Str)
tr.step()

