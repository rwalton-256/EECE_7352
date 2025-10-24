import simulator
import numpy as np
import matplotlib.pyplot as plt

# This is a test using the MATMULTIPLY kernel.

num_proc_blocks_opt = np.arange(70)+1

instr = simulator.parse_instructions('kernels.sasm','_Z6matmulPfS_S_')
cmem = bytearray( 1024 )

ARR_DIM = 64
ARR_SIZE = ARR_DIM * ARR_DIM
mat_a = np.random.randn( ARR_SIZE ).astype( np.float32 )
mat_b = np.random.randn( ARR_SIZE ).astype( np.float32 )

cycles_vec = []
for num_proc_blocks in num_proc_blocks_opt:
    sm = simulator.SM(
        aNumSchedulers=num_proc_blocks,
        aMemorySize=1024*1024,
        aWarpWidth=32,
        aWarpInterleave=4,
        aIntLatency=2,
        aFloatLatency=3,
        aMemLatency=3,
        aNumReg=32,
        aMultiIssue=True
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
    print(cycles)

plt.plot( num_proc_blocks_opt, cycles_vec )

plt.ylabel("Cycles")
plt.xlabel("Processing Blocks in SM")
plt.title("Total Cycles Required to Compute Mat-Multiply Workload")
plt.show()

