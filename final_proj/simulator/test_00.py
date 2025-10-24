import simulator
import numpy as np
import time

# This is a test using the VECADD kernel.

instr = simulator.parse_instructions('kernels.sasm','_Z6vecmulPfS_')
cmem = bytearray( 1024 )

sm = simulator.SM(
    aNumSchedulers=1,
    aMemorySize=1024*1024,
    aWarpWidth=32,
    aWarpInterleave=4,
    aIntLatency=2,
    aFloatLatency=4,
    aMemLatency=4,
    aNumReg=16,
    aMultiIssue=True
)

sm.exec_kernel(
    32,
    32*6,
    aInstructions=instr,
    aConstMem=cmem,
    aArgs=[
        np.arange( 256 ).astype( np.float32 ),
        np.arange( 256 ).astype( np.float32 )
    ]
)

while not sm.idle():
    input()
    print(simulator.CLS)
    sm.print()
    sm.clock()
    #stime.sleep( 0.75 )

