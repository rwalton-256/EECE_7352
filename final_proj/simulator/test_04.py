import simulator
import numpy as np
import matplotlib.pyplot as plt

# This is a test using the MATMUL kernel.

ARR_DIM = 64
ARR_SIZE = ARR_DIM * ARR_DIM

multi_issue_opt = [True,False]
warp_interleave_opt = [1,2,3,4]
warp_width_opt = [1,2,4,8,32]
num_scheduler_opt = [1,2,3,4]

instr = simulator.parse_instructions('kernels.sasm','_Z6matmulPfS_S_')
cmem = bytearray( 1024 )

mat_a = np.random.randn( ARR_SIZE ).astype( np.float32 )
mat_b = np.random.randn( ARR_SIZE ).astype( np.float32 )
mul_truth = np.linalg.matmul( mat_a.reshape((ARR_DIM,ARR_DIM)), mat_b.reshape((ARR_DIM,ARR_DIM)) )

for multi_issue in multi_issue_opt:
    cycles_vec = []
    for warp_interleave in warp_interleave_opt:
        for warp_width in warp_width_opt:
            for num_scheduler in num_scheduler_opt:
                sm = simulator.SM(
                    aNumSchedulers=num_scheduler,
                    aMemorySize=1024*1024,
                    aWarpWidth=warp_width,
                    aWarpInterleave=warp_interleave,
                    aIntLatency=3,
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
                    cycles += 1

                res = np.frombuffer( sm.mMemory[ARR_SIZE*4*2:ARR_SIZE*4*3], dtype=np.float32 ).reshape((ARR_DIM,ARR_DIM))
                diff = np.abs( res-mul_truth )
                assert np.max(diff) < 1e-5

                print(f"Validated config: multi-issue: {'yes' if multi_issue else 'no'}, warp interleave: {warp_interleave}, warp width: {warp_width}, num scheduler: {num_scheduler}, cycles: {cycles}")

print(f"Mat Mul unit test passed successfully!")

