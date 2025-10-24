import simulator
import numpy as np
import matplotlib.pyplot as plt

# This is a test using the VECADD kernel.

multi_issue_opt = [True,False]
warp_interleave_opt = [1,2,3,4,5,6]

instr = simulator.parse_instructions('kernels.sasm','_Z6vecmulPfS_')
cmem = bytearray( 1024 )

for multi_issue in multi_issue_opt:
    cycles_vec = []
    for warp_interleave in warp_interleave_opt:
        sm = simulator.SM(
            aNumSchedulers=2,
            aMemorySize=1024*1024,
            aWarpWidth=32,
            aWarpInterleave=warp_interleave,
            aIntLatency=2,
            aFloatLatency=3,
            aMemLatency=3,
            aNumReg=16,
            aMultiIssue=multi_issue
        )

        sm.exec_kernel(
            256,
            8192,
            aInstructions=instr,
            aConstMem=cmem,
            aArgs=[
                np.arange( 8192 ).astype( np.float32 ),
                np.arange( 8192 ).astype( np.float32 )
            ]
        )

        cycles = 0
        while not sm.idle():
            sm.clock()
            cycles = cycles+1
        cycles_vec.append( cycles )

    plt.plot( warp_interleave_opt, cycles_vec )

plt.legend(["Multi-Issue Enabled","Multi-Issue Disabled"])
plt.ylabel("Cycles")
plt.xlabel("Num Simultaneous Warps Interleaved")
plt.title("Total Cycles Required to Compute Vec-Multiply Workload")
plt.show()


