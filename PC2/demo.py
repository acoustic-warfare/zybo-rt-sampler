from lib.beamformer import *

import sys

if sys.argv[1] == "miso":
    miso()
elif sys.argv[1] == "mimo":
    mimo()
else:
    print("invalid argument")

# mimo()