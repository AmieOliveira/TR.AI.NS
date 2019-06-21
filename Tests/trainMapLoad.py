#!/usr/bin/python

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from Train import Train

pos0 = (0, 0)
tr = Train(1, pos0, "../mapFile", log=True)