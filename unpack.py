# Python unpacker for World Cup 1998 website logs

import struct
import json
import sys

fmt = "!IIIIcccc"
size = struct.calcsize(fmt) # 20

f = open(sys.argv[1])
output = []
s = f.read(size)

count = 0
outfile = open(sys.argv[2], 'w')

while len(s) != 0:
  if len(s) == size:
    x = struct.unpack(fmt, s)
    # y = (x[0], x[1], x[2], x[3], ord(x[4]), ord(x[5]), ord(x[6]), ord(x[7]))
    y = list(x[:4]) + [ord(p) for p in x[4:]]

    j = json.dumps(y, outfile)
    outfile.write(j + "\n")
  s = f.read(size)
  count += 1
  f.seek(count * size, 0)
