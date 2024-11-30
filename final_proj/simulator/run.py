import kern_mem
import typing
import time
import re
import typing
import enum
import struct

class colors:
    DEF='\033[37m'
    RED='\033[31m'
    GRN='\033[32m'
    YEL='\033[33m'
    BLU='\033[34m'
    MAG='\033[35m'
    CYA='\033[36m'

clist = [
    colors.RED,
    colors.GRN,
    colors.YEL,
    colors.BLU,
    colors.MAG,
    colors.CYA
]

CLS='\033[2J'

mem = kern_mem.ram_buffer( 1024*1024 )

cmem = kern_mem.ram_buffer( 1024 )
#cmem.allocate( "reserved", 0x160 )

# Allocate memory in global memory
#mem.allocate( "buf", 4*128*32 )
# Allocate pointer in constant memory for the kernel parameter,
# a pointer to global memory
#cmem.allocate( "pbuf", 4 )
# Write the pointer to the constant memory
#cmem.write( 'pbuf', 'I', 0, mem.get_off('buf') )

#print( hex(cmem.read( 'pbuf', 'I', 0) ) )
#print(cmem.mBuffer)

class Work_Item:
    mIdx : int
    mNumReg : int
    mRegs : bytearray
    def __init__(
        self,
        aIdx : int,
        aNumReg : int
    ):
        self.mIdx = aIdx
        self.mNumReg = aNumReg
        self.mRegs = bytearray( 4 * aNumReg )

    def read_reg( self, aReg : int, aType : str = 'I' ):
        return struct.unpack( aType, self.mRegs[4*aReg:4*(aReg+1)] )[0]

    def write_reg( self, aReg : int, aVal : typing.Any, aType : str = 'I' ):
        self.mRegs[4*aReg:4*(aReg+1)] = struct.pack( aType, aVal )

class SrcOperand:
    class Tp( enum.Enum ):
        REG=0
        CONST=1
        IMM=2

    mTp : Tp
    mVal : int | str

    def __init__( self, aOperand : str ):
        if aOperand[0] == 'R':
            self.mTp = self.Tp.REG
            self.mVal = int( aOperand[1:] )
        elif aOperand[0] == 'c':
            self.mTp = self.Tp.CONST
            self.mVal = int( aOperand.split(']')[-2].split('[')[-1], base=16 )
        else:
            self.mTp = self.Tp.IMM
            self.mVal = int( aOperand, base=16 )

    def get_val( self, aWorkItem : Work_Item, aConstMem : bytearray, aType : int = 'I' ):
        match self.mTp:
            case self.Tp.REG:
                return aWorkItem.read_reg( self.mVal, aType=aType )
            case SrcOperand.Tp.CONST:
                return struct.unpack(
                    aType,
                    aConstMem[self.mVal:self.mVal+4]
                )[0]
            case self.Tp.IMM:
                return self.mVal
            case _:
                assert 0

class Hazard:
    class Tp( enum.Enum ):
        RAW=0
    mType : Tp
    mReg : int
    def __init__(
        self,
        aType : str,
        aReg : int
    ):
        self.mType = aType
        self.mReg = aReg

    def __eq__( self ):
        print("HERE")

    def __neq__( self ):
        print("HERE")

class Instruction:
    class Op( enum.Enum ):
        MOV=0
        S2R=1
        SHR=2
        ISCADD=3
        IADD=4
        MOV32I=5
        STG=6
        NOP=7
        EXIT=8
        BRA=9
        FMA=10

    class MovTp( enum.Enum ):
        REG=0
        CONST=1
        IMM=2

    class Datapath( enum.Enum ):
        INT=0
        FLT=1
        MEM=2

    mOp : Op
    mDest : int
    mSrc : typing.List[ SrcOperand ] | str
    mLabel : str
    mDatapath : Datapath
    def __init__( self, aInst : str, aLabel = "" ):
        self.mOp = self.Op[ aInst.split(" ")[0].split('.')[0] ]

        operands = [ s.strip().split(".")[0] for s in " ".join(aInst.split(";")[0].split(" ")[1:]).split(",") ]

        # Unpack destination register if applicable
        match self.mOp:
            case (
                self.Op.MOV | self.Op.S2R | self.Op.SHR | self.Op.ISCADD | self.Op.IADD |
                self.Op.MOV32I
            ):
                self.mDest = int( operands[0][1:] )

        # Register source operands
        match self.mOp:
            case self.Op.S2R:
                self.mSrc = operands[1]

            # Typical instruction with source operands
            case (
                self.Op.MOV | self.Op.SHR | self.Op.ISCADD | self.Op.IADD | self.Op.MOV32I
            ):
                self.mSrc = [ SrcOperand( s ) for s in operands[1:] ]

        # Select datapath type
        match self.mOp:
            case (
                self.Op.MOV | self.Op.SHR | self.Op.ISCADD | self.Op.IADD | self.Op.S2R
            ):
                self.mDatapath = self.Datapath.INT
            case (
                self.Op.FMA
            ):
                self.mDatapath = self.Datapath.FLT
            case (
                self.Op.STG
            ):
                self.mDatapath = self.Datapath.MEM

        # Set the label
        self.mLabel = aLabel

    def __str__( self ):
        return f"Instruction, type: {self.mType}, operands: {self.mOperands}, label: {self.mLabel}"

    def __repr__( self ):
        return self.__str__()

    def get_hazards(
            self,
        ) -> typing.List[ Hazard ]:
        hazards = []

        match self.mOp:
            case (
                self.Op.MOV | self.Op.S2R | self.Op.SHR | self.Op.ISCADD | self.Op.IADD |
                self.Op.MOV32I
            ):
                hazards.append( Hazard( Hazard.Tp.RAW, self.mDest ) )

        return hazards

def parse_instructions( aFilename:str, aKernelName:str ) -> typing.List[ Instruction ]:
    text = open(aFilename).read()

    kernel = text.split( f".text.{aKernelName}:" )[-1]
    # remove all occurrences streamed comments (/*COMMENT */) from string
    kernel = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,kernel)
    # remove all occurrence single-line comments (//COMMENT\n ) from string
    kernel = re.sub(re.compile("//.*?\n" ) ,"" ,kernel)
    kernel = kernel.strip()
    kernel = kernel.replace("{","")
    kernel = kernel.replace("}",";")

    instructions = []
    cur_label = ""
    while kernel != "":
        if kernel[0] == '.':
            cur_label = kernel.split(".")[1].split(":")[0]
            kernel = ":".join(kernel.split(':')[1:]).strip()
        else:
            op = kernel.split(";")[0]
            instructions.append( Instruction( op, cur_label ) )
            cur_label = ""
            kernel = ";".join(kernel.split(";")[1:]).strip()
    
    return instructions

class Warp:
    mWarpWidth : int
    mWorkItems : typing.List[ Work_Item ]
    mInstructions : typing.List[ Instruction ]
    mHazards : typing.List[ Hazard ]
    mIptr : int
    mIntLatency : int
    mFloatLatency : int
    mMemLatency : int
    mColor : str
    mConstMem : bytearray
    mWarpIdx : int
    mNumWarp : int
    mGroupIdx : int
    mNumGroup : int

    def __init__(
        self,
        aWarpWidth : int,
        aWarpIdx : int,
        aNumWarp : int,
        aGroupIdx : int,
        aNumGroup : int,
        aNumReg : int,
        aInstructions : typing.List[ Instruction ],
        aConstMem : bytearray
    ):
        self.mWarpWidth = aWarpWidth
        self.mWorkItems = []
        for i in range( self.mWarpWidth ):
            self.mWorkItems.append( Work_Item( aIdx=i, aNumReg=aNumReg ) )
        self.mInstructions = aInstructions
        self.mIptr = 0
        self.mHazards = []
        self.mConstMem = aConstMem

        self.mWarpIdx = aWarpIdx
        self.mNumGroup = aNumWarp
        self.mGroupIdx = aGroupIdx
        self.mNumGroup = aNumGroup

    def pop_instr( self ) -> Instruction | None:
        instr = self.mInstructions[ self.mIptr ]
        for hazard in self.mHazards:
            pass
        self.mIptr += 1

        hazards = instr.get_hazards(
        )
        for hazard in hazards:
            self.mHazards.append( hazard )

        return instr

    def clock( self ):
        for hazard in self.mHazards:
            hazard.clock()

        self.mHazards.remove( 'stale' )

    def set_color( self, aColor : str ):
        self.mColor = aColor
    
    def get_color( self ) -> str:
        return self.mColor

class Int_Datapath:
    mIntLatency : int

    mNxt : typing.Tuple[ Instruction, Warp ] | None
    mPipeline : typing.List[
        typing.Tuple[ Instruction, Warp ] | None
    ]

    def __init__(
        self,
        aIntLatency : int
    ):
        self.mIntLatency = aIntLatency

        self.mNxt = None
        self.mPipeline = self.mIntLatency * [ None ]

    def clock( self ):
        if self.mPipeline[-1]:
            instruction,warp = self.mPipeline[-1]

            for item in warp.mWorkItems:
                match instruction.mOp:
                    case Instruction.Op.MOV:
                        val = instruction.mSrc[0].get_val(
                            item,
                            warp.mConstMem
                        )
                        print(f"MOV: Writing {val} to work item {item.mIdx}")
                        item.write_reg(
                            instruction.mDest,
                            val
                        )
                    case Instruction.Op.S2R:
                        match instruction.mSrc:
                            case "SR_TID":
                                    val = warp.mWarpIdx * warp.mWarpWidth + item.mIdx
                            case _:
                                assert 0
                        print(f"S2R: Writing {val} to work item {item.mIdx}")
                        item.write_reg(
                            instruction.mDest,
                            val
                        )
                    case Instruction.Op.SHR:
                        # Assumed Register Operand
                        op_a = instruction.mSrc[0].get_val(
                            item,
                            warp.mConstMem
                        )
                        # 
                        op_b = instruction.mSrc[1].get_val(
                            item,
                            warp.mConstMem
                        )
                        val = op_a >> op_b
                        print(f"SHR: Writing {val} to work item {item.mIdx}")
                        item.write_reg(
                            instruction.mDest,
                            val
                        )
                    case _:
                        raise RuntimeError( f"Unimplemented instruction: {instruction.mOp.name}" )
                

        for i in range( self.mIntLatency - 1 ):
            self.mPipeline[ self.mIntLatency - i - 1 ] = self.mPipeline[ self.mIntLatency - i - 2 ]

        self.mPipeline[0] = self.mNxt

        self.mNxt = None

    def print( self, prefix='' ):
        int_str = "|"
        for i in range( self.mIntLatency ):
            if self.mPipeline[i]:
                instr,warp = self.mPipeline[i]
                int_str += f"{warp.get_color()}**{colors.DEF}|"
            else:
                int_str += f"  |"
        print(f"{prefix} INT  {int_str}")

    def push( self, aInstruction : Instruction, aWarp : Warp ):
        self.mNxt = (aInstruction, aWarp)


class Float_Datapath:
    mWarpWidth : int
    mFloatLatency : int
    def __init__(
        self,
        aWarpWidth : int,
        aFloatLatency : int
    ):
        self.mWarpWidth = aWarpWidth
        self.mFloatLatency = aFloatLatency

    def clock( self ):
        pass

    def print( self, prefix='' ):
        flt_str = "|"
        for i in range( self.mFloatLatency ):
            flt_str += f"C{i}|"
        print(f"{prefix} FLT  {flt_str}")

    def push( self, aInstruction : Instruction, aWarp : Warp ):
        pass

class Memory_Datapath:
    mWarpWidth : int
    mMemLatency : int
    def __init__(
        self,
        aWarpWidth : int,
        aMemLatency : int
    ):
        self.mWarpWidth = aWarpWidth
        self.mMemLatency = aMemLatency

    def clock( self ):
        pass

    def print( self, prefix='' ):
        mem_str = "|"
        for i in range( self.mMemLatency ):
            mem_str += f"C{i}|"
        print(f"{prefix} MEM  {mem_str}")

    def push( self, aInstruction : Instruction, aWarp : Warp ):
        pass

class Scheduler:
    mWarpWidth : int
    mWarpInterleave : int
    mIntLatency : int
    mFloatLatency : int
    mIntDatapath : Int_Datapath
    mFloatDatapath : Float_Datapath
    mMemDatapath : Memory_Datapath

    mActiveWarps : typing.List[ Warp ]
    mWarpIdx : int
    mCidx : int

    def __init__(
        self,
        aWarpWidth : int,
        aWarpInterleave : int,
        aIntLatency : int,
        aFloatLatency : int,
        aMemLatency : int
    ):
        self.mWarpWidth = aWarpWidth
        self.mWarpInterleave = aWarpInterleave
        self.mIntLatency = aIntLatency
        self.mFloatLatency = aFloatLatency

        self.mIntDatapath = Int_Datapath( aIntLatency=aIntLatency )
        self.mFloatDatapath = Float_Datapath( aWarpWidth=aWarpWidth, aFloatLatency=aFloatLatency )
        self.mMemDatapath = Memory_Datapath( aWarpWidth=aWarpWidth, aMemLatency=aMemLatency )

        self.mActiveWarps = []
        self.mWarpIdx = 0
        self.mCidx = 0

    def at_capacity( self ) -> bool:
        return len( self.mActiveWarps ) == self.mWarpInterleave

    def insert_warp( self, aWarp : Warp ):
        assert len( self.mActiveWarps ) < self.mWarpInterleave
        aWarp.set_color( clist[self.mCidx] )
        self.mCidx += 1
        self.mActiveWarps.append( aWarp )

    def clock( self ):
        for i in range( len( self.mActiveWarps ) ):
            warp = self.mActiveWarps[ ( i + self.mWarpIdx ) % len( self.mActiveWarps ) ]
            instr = warp.pop_instr()

            if instr == None:
                continue

            match instr.mDatapath:
                case Instruction.Datapath.INT:
                    self.mIntDatapath.push( instr, warp )
                case Instruction.Datapath.FLT:
                    self.mFloatDatapath.push( instr, warp )
                case Instruction.Datapath.MEM:
                    self.mMemDatapath.push( instr, warp )

            self.mWarpIdx = ( self.mWarpIdx + 1 ) % len( self.mActiveWarps )

            break

        self.mIntDatapath.clock()
        self.mFloatDatapath.clock()

    def print( self, prefix='' ):
        print(f"{prefix}{48*'-'}")
        prefix = prefix + "| "

        slot_str = "|"
        for i in range( self.mWarpInterleave ):
            if i < len( self.mActiveWarps ):
                slot_str += f"{self.mActiveWarps[i].get_color()}***{colors.DEF}|"
            else:
                slot_str += "   |"


        print(f"{prefix} Warp Scheduler     {slot_str}")
        print(f"{prefix}                    {(self.mWarpInterleave*4+1)*'-'}")
        print(f"{prefix}")

        self.mIntDatapath.print( prefix=prefix )

        print(f"{prefix}")

        self.mFloatDatapath.print( prefix=prefix )

        print(f"{prefix}")

        self.mMemDatapath.print( prefix=prefix )

        print(f"{prefix}")

        prefix = prefix + '| '

class SM:
    mNumSchedulers : int
    mSchedulers : typing.List[ Scheduler ]
    mSharedMemSize : int
    mReadyWarps : typing.List[ Warp ]
    mWarpWidth : int
    mNumReg : int
    mIntLatency : int
    mFloatLatency : int
    mMemLatency : int

    def __init__(
        self,
        aNumSchedulers : int,
        aSharedMemSize : int,
        aWarpWidth : int,
        aWarpInterleave : int,
        aIntLatency : int,
        aFloatLatency : int,
        aMemLatency : int,
        aNumReg : int
    ):
        self.mNumSchedulers = aNumSchedulers
        self.mSchedulers = []
        for i in range( self.mNumSchedulers ):
            self.mSchedulers.append(
                Scheduler(
                    aWarpWidth=aWarpWidth,
                    aWarpInterleave=aWarpInterleave,
                    aIntLatency=aIntLatency,
                    aFloatLatency=aFloatLatency,
                    aMemLatency=aMemLatency
                )
            )
        self.mWarpWidth = aWarpWidth
        self.mNumReg = aNumReg
        self.mReadyWarps = []
        self.mIntLatency = aIntLatency
        self.mFloatLatency = aFloatLatency
        self.mMemLatency = aMemLatency

    def exec_kernel(
        self,
        aGroupSize : int,
        aGlobalSize : int,
        aInstructions : typing.List[ Instruction ],
        aConstMem : bytearray
    ):
        # Group Size must be a multiple of warp width
        assert aGroupSize % self.mWarpWidth == 0
        num_warp = aGroupSize // self.mWarpWidth

        assert aGlobalSize % aGroupSize == 0
        num_group = aGlobalSize // aGroupSize

        for i_group in range( num_group ):
            for i_warp in range( num_warp ):
                self.mReadyWarps.append(
                    Warp(
                        self.mWarpWidth,
                        aWarpIdx=i_warp,
                        aNumWarp=num_warp,
                        aGroupIdx=i_group,
                        aNumGroup=num_group,
                        aNumReg=self.mNumReg,
                        aInstructions=aInstructions,
                        aConstMem=aConstMem
                    )
                )

    def clock( self ):
        # Insert warps into the schedulers in a round robin fashion
        for scheduler in self.mSchedulers:
            if not scheduler.at_capacity():
                scheduler.insert_warp( self.mReadyWarps.pop( 0 ) )

        for scheduler in self.mSchedulers:
            scheduler.clock()

    def print( self, prefix='' ):
        print(f"{prefix}{50*'-'}")
        prefix = prefix + '| '
        print(f"{prefix} Streaming Multiprocessor")
        #print(f"{prefix}   queued warps: {self.mReadyWarps.count()}")
        for scheduler in self.mSchedulers:
            scheduler.print( prefix=prefix )
        print(f"{prefix}{48*'-'}")
        prefix = prefix[:-2]
        print(f"{prefix}{50*'-'}")

sm = SM(
    aNumSchedulers=4,
    aSharedMemSize=1024*1024,
    aWarpWidth=4,
    aWarpInterleave=4,
    aIntLatency=3,
    aFloatLatency=3,
    aMemLatency=3,
    aNumReg=16
)

print(f"{CLS}")
sm.print()

instr = parse_instructions('foo.sasm','_Z6KernelPj')

cmem = bytearray( 1024 )
sm.exec_kernel( 32, 64, aInstructions=instr, aConstMem=cmem )

for i in range(100):
#    time.sleep(1)

    sm.clock()

    #print(f"{CLS}")
    sm.print()

exit()
























iptr = 0
while True:
    instruction = instr[iptr]

    print(f"- {iptr:03d} {instruction.mType}")

    if instruction.mType == "MOV":
        dst = instruction.mOperands[0]

        if instruction.mOperands[1][0:6] == 'c[0x0]':
            val = cmem.read( 'I', int( instruction.mOperands[1].split('[')[-1].split(']')[0], base=16 ) )

        print(f"  Storing {val:08x} to {dst}")
        regs[dst] = val

    elif instruction.mType == "S2R":
        dst = instruction.mOperands[0]

        if instruction.mOperands[1] == 'SR_TID.X':
            val = 0
        
        print(f"  Storing {val:08x} to {dst}")

    elif instruction.mType == "SHR.U32":
        dst = instruction.mOperands[0]



    iptr += 1
