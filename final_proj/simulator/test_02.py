import simulator
import numpy as np
import matplotlib.pyplot as plt

# This is a test using the VECADD kernel.

GLOBAL_SIZE = 2048
LOCAL_SIZE = 256

multi_issue_opt = [True,False]
warp_interleave_opt = [1,2,3,4]
warp_width_opt = [1,2,4,8,32]
num_scheduler_opt = [1,2,3,4]

instr = simulator.parse_instructions('kernels.sasm','_Z6vecmulPfS_')
cmem = bytearray( 1024 )

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
                    aNumReg=16,
                    aMultiIssue=multi_issue
                )

                sm.exec_kernel(
                    LOCAL_SIZE,
                    GLOBAL_SIZE,
                    aInstructions=instr,
                    aConstMem=cmem,
                    aArgs=[
                        np.arange( GLOBAL_SIZE ).astype( np.float32 ),
                        np.arange( GLOBAL_SIZE ).astype( np.float32 )
                    ]
                )

                cycles = 0
                while not sm.idle():
                    sm.clock()
                    cycles += 1

                res = np.frombuffer( sm.mMemory[0:GLOBAL_SIZE*4], dtype=np.float32 )
                diff = np.arange( GLOBAL_SIZE ).astype( np.float32 ) ** 2 - res
                assert np.max(diff) == 0

                print(f"Validated config: multi-issue: {'yes' if multi_issue else 'no'}, warp interleave: {warp_interleave}, warp width: {warp_width}, num scheduler: {num_scheduler}, cycles: {cycles}")

print(f"Vec-Mult unit test passed successfully!")
