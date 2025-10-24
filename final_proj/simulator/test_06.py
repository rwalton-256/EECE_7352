import simulator
import numpy as np
import matplotlib.pyplot as plt

# This is a test using the MATMULTIPLY kernel.

multi_issue_opt = [True,False]
warp_interleave_opt = [1,2,3,4,5,6]

instr = simulator.parse_instructions('kernels.sasm','_Z6matmulPfS_S_')
cmem = bytearray( 1024 )

ARR_DIM = 64
ARR_SIZE = ARR_DIM * ARR_DIM
mat_a = np.random.randn( ARR_SIZE ).astype( np.float32 )
mat_b = np.random.randn( ARR_SIZE ).astype( np.float32 )

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
            aNumReg=32,
            aMultiIssue=multi_issue
        )

        sm.exec_kernel(
            32,
            ARR_SIZE,
            aInstructions=instr,
            aConstMem=cmem,
            aArgs=[
                mat_a,
                mat_b,
                np.zeros( ARR_SIZE ).astype( np.float32 )
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
plt.title("Total Cycles Required to Compute Mat-Multiply Workload")
plt.show()

