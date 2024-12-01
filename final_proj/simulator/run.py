import kern_mem
import typing
import time
import re
import typing
import enum
import struct
import numpy as np
import matplotlib.pyplot as plt

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

def hexdump( bb : bytearray, perline : int = 8, sz : int = 4 ):
    for i_line in range( len(bb) // ( perline * sz ) ):
        sl = bb[i_line * perline * sz : (i_line + 1) * perline * sz]
        print(f"{i_line*sz*perline:04x} {' '.join( [ sl[i*sz:(i+1)*sz].hex() for i in range(perline) ] )}")

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
        ZERO=3

    mTp : Tp
    mVal : int | str
    mMod : str

    def __init__( self, aOperand : str ):
        if aOperand[0:2] == 'RZ':
            self.mTp = self.Tp.ZERO
        elif aOperand[0] == 'R':
            self.mTp = self.Tp.REG
            split = aOperand.split('.')
            self.mVal = int( split[0][1:] )
            if len( split ) > 1:
                self.mMod = split[1]
            else:
                self.mMod = ''
        elif aOperand[0] == 'c':
            self.mTp = self.Tp.CONST
            self.mVal = int( aOperand.split(']')[-2].split('[')[-1], base=16 )
            split = aOperand.split('.')
            if len( split ) > 1:
                self.mMod = split[1]
            else:
                self.mMod = ''
        else:
            self.mTp = self.Tp.IMM
            self.mVal = int( aOperand, base=16 )

    def get_val( self, aWorkItem : Work_Item, aConstMem : bytearray, aType : int = 'I' ):
        match self.mTp:
            case self.Tp.REG:
                val = aWorkItem.read_reg( self.mVal, aType=aType )
                if self.mMod == 'H1':
                    val = val >> 16
                return val
            case SrcOperand.Tp.CONST:
                val = struct.unpack(
                    aType,
                    aConstMem[self.mVal:self.mVal+4]
                )[0]
                if self.mMod == 'H1':
                    val = val >> 16
                return val
            case self.Tp.IMM:
                return self.mVal
            case self.Tp.ZERO:
                return 0
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
        SHL=11
        LDG=12
        FMUL=13
        XMAD=14

    class MovTp( enum.Enum ):
        REG=0
        CONST=1
        IMM=2

    class Datapath( enum.Enum ):
        INT=0
        FLT=1
        MEM=2

    mOp : Op
    mDst : int
    mSrc : typing.List[ SrcOperand ] | str
    mLabel : str
    mCarry : str
    mDatapath : Datapath
    def __init__( self, aInst : str, aLabel = "" ):
        self.mOp = self.Op[ aInst.split(" ")[0].split('.')[0] ]
        self.mMods = aInst.split(" ")[0].split('.')[1:]

        operands = [ s.strip() for s in " ".join(aInst.split(";")[0].split(" ")[1:]).split(",") ]

        # Unpack destination register if applicable
        match self.mOp:
            case (
                self.Op.MOV | self.Op.S2R | self.Op.SHR | self.Op.ISCADD | self.Op.IADD |
                self.Op.MOV32I | self.Op.SHL | self.Op.LDG | self.Op.XMAD | self.Op.FMUL
            ):
                self.mDst = int( operands[0][1:].split('.')[0] )
                split = operands[0][1:].split('.')
                self.mCarry = False
                if len( split ) > 1:
                    if split[1] == "CC":
                        self.mCarry = True
            case (
                self.Op.STG
            ):
                # Strip off the brackets for the memory operand
                self.mDst = int( operands[0][2:-1] )

        # Register source operands
        match self.mOp:
            case self.Op.S2R:
                self.mSrc = operands[1]

            # Typical instruction with source operands
            case (
                self.Op.MOV | self.Op.SHR | self.Op.ISCADD | self.Op.IADD | self.Op.MOV32I |
                self.Op.STG | self.Op.SHL | self.Op.XMAD | self.Op.FMUL
            ):
                self.mSrc = [ SrcOperand( s ) for s in operands[1:] ]
            case (
                self.Op.LDG
            ):
                # Strip off the brackets for the memory operand
                self.mSrc = [ SrcOperand( s[1:-1] ) for s in operands[1:] ]

        # Select datapath type
        match self.mOp:
            case (
                self.Op.MOV | self.Op.SHR | self.Op.ISCADD | self.Op.IADD | self.Op.S2R |
                self.Op.MOV32I | self.Op.SHL | self.Op.XMAD
            ):
                self.mDatapath = self.Datapath.INT
            case (
                self.Op.FMA | self.Op.FMUL
            ):
                self.mDatapath = self.Datapath.FLT
            case (
                self.Op.STG | self.Op.LDG
            ):
                self.mDatapath = self.Datapath.MEM
            case _:
                self.mDatapath = None

        # Set the label
        self.mLabel = aLabel

    def get_hazards(
            self,
        ) -> typing.List[ Hazard ]:
        hazards = []

        match self.mOp:
            case (
                self.Op.MOV | self.Op.S2R | self.Op.SHR | self.Op.ISCADD | self.Op.IADD |
                self.Op.MOV32I
            ):
                hazards.append( Hazard( Hazard.Tp.RAW, self.mDst ) )

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
    mCarrySet : bool

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
        self.mConstMem = aConstMem

        self.mWarpIdx = aWarpIdx
        self.mNumGroup = aNumWarp
        self.mGroupIdx = aGroupIdx
        self.mNumGroup = aNumGroup
        self.mCarrySet = False

    def peek_instr( self ) -> Instruction:
        return self.mInstructions[ self.mIptr ]

    def pop_instr( self ):
        self.mIptr += 1

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
                    case ( Instruction.Op.MOV | Instruction.Op.MOV32I ):
                        val = instruction.mSrc[0].get_val(
                            item,
                            warp.mConstMem
                        )
                    case Instruction.Op.S2R:
                        match instruction.mSrc:
                            case "SR_TID.X":
                                val = warp.mWarpIdx * warp.mWarpWidth + item.mIdx
                            case "SR_CTAID.X":
                                val = warp.mGroupIdx
                            case _:
                                assert 0
                    case Instruction.Op.SHR:
                        op_a,op_b = [
                            op.get_val( item, warp.mConstMem ) for op in instruction.mSrc
                        ]
                        val = op_a >> op_b
                    case Instruction.Op.ISCADD:
                        op_a,op_b,op_c = [
                            op.get_val( item, warp.mConstMem ) for op in instruction.mSrc
                        ]
                        val = op_b + op_a << op_c
                    case Instruction.Op.IADD:
                        op_a,op_b = [
                            op.get_val( item, warp.mConstMem ) for op in instruction.mSrc
                        ]
                        val = op_a + op_b
                        if instruction.mMods == "X":
                            if warp.mCarrySet:
                                val += 1
                        if instruction.mCarry and val > 0xffffffff:
                            warp.mCarrySet = True
                        val = val & 0xffffffff
                    case Instruction.Op.SHL:
                        op_a,op_b = [
                            op.get_val( item, warp.mConstMem ) for op in instruction.mSrc
                        ]
                        val = op_a << op_b
                    case Instruction.Op.XMAD:
                        op_a,op_b,op_c = [
                            op.get_val( item, warp.mConstMem ) for op in instruction.mSrc
                        ]
                        product = ( op_a & 0xffff ) * ( op_b & 0xffff )

                        # XMAD reverse engineering discussed here: https://forums.developer.nvidia.com/t/xmad-meaning/46653
                        if "MRG" in instruction.mMods:
                            val = ( ( product + op_c ) & 0xffff ) | ( ( op_b & 0xffff ) << 16 )
                        elif "PSL" in instruction.mMods and "CBCC" in instruction.mMods:
                            val = ( product << 16 ) + op_c + ( ( op_b & 0xffff ) << 16 )
                        else:
                            val = product + op_c
                    case _:
                        raise RuntimeError( f"Unimplemented instruction: {instruction.mOp.name}" )

                #print(f"{instruction.mOp.name}: Writing {val} to work item {item.mIdx}")
                item.write_reg(
                    instruction.mDst,
                    val
                )

        for i in range( self.mIntLatency - 1 ):
            self.mPipeline[ self.mIntLatency - i - 1 ] = self.mPipeline[ self.mIntLatency - i - 2 ]

        self.mPipeline[0] = self.mNxt

        self.mNxt = None

    def print( self, prefix='' ):
        int_str = "|"
        for i in range( self.mIntLatency ):
            if self.mPipeline[i]:
                instr,warp = self.mPipeline[i]
                int_str += f"{warp.get_color()}{instr.mOp.name:6}{colors.DEF}|"
            else:
                int_str += f"{6*' '}|"
        print(f"{prefix} INT  {int_str}")

    def is_hazard( self, aWarp : Warp, aHazRegs : typing.List[ int ] ) -> bool:
        for i in range( self.mIntLatency - 1 ):
            if self.mPipeline[i] is None:
                continue
            instr,warp = self.mPipeline[i]

            if warp != aWarp:
                continue

            # Check for hazards between instructions in the pipeline
            # and registers associated with the instruction to be added
            for reg in aHazRegs:
                if reg == instr.mDst:
                    return True
        return False

    def push( self, aInstruction : Instruction, aWarp : Warp ):
        self.mNxt = (aInstruction, aWarp)

    def idle( self ):
        for i in range( self.mIntLatency ):
            if self.mPipeline[i]:
                return False
        return True

class Float_Datapath:
    mFloatLatency : int

    mNxt : typing.Tuple[ Instruction, Warp ] | None
    mPipeline : typing.List[
        typing.Tuple[ Instruction, Warp ] | None
    ]

    def __init__(
        self,
        aFloatLatency : int
    ):
        self.mFloatLatency = aFloatLatency
        self.mNxt = None
        self.mPipeline = self.mFloatLatency * [ None ]

    def clock( self ):
        if self.mPipeline[-1]:
            instruction,warp = self.mPipeline[-1]

            for item in warp.mWorkItems:
                match instruction.mOp:
                    case Instruction.Op.FMUL:
                        op_a,op_b = [
                            op.get_val( item, warp.mConstMem, 'f' ) for op in instruction.mSrc
                        ]
                        val = op_a * op_b
                    case _:
                        raise RuntimeError( f"Unimplemented instruction: {instruction.mOp.name}" )

                #print(f"{instruction.mOp.name}: Writing {val} to work item {item.mIdx}")
                item.write_reg(
                    instruction.mDst,
                    val,
                    'f'
                )

        for i in range( self.mFloatLatency - 1 ):
            self.mPipeline[ self.mFloatLatency - i - 1 ] = self.mPipeline[ self.mFloatLatency - i - 2 ]

        self.mPipeline[0] = self.mNxt

        self.mNxt = None

    def print( self, prefix='' ):
        int_str = "|"
        for i in range( self.mFloatLatency ):
            if self.mPipeline[i]:
                instr,warp = self.mPipeline[i]
                int_str += f"{warp.get_color()}{instr.mOp.name:6}{colors.DEF}|"
            else:
                int_str += f"{6*' '}|"
        print(f"{prefix} FLT  {int_str}")

    def is_hazard( self, aWarp : Warp, aHazRegs : typing.List[ int ] ) -> bool:
        for i in range( self.mFloatLatency - 1 ):
            if self.mPipeline[i] is None:
                continue
            instr,warp = self.mPipeline[i]

            if warp != aWarp:
                continue

            # Check for hazards between instructions in the pipeline
            # and registers associated with the instruction to be added
            for reg in aHazRegs:
                if reg == instr.mDst:
                    return True
        return False

    def push( self, aInstruction : Instruction, aWarp : Warp ):
        self.mNxt = (aInstruction, aWarp)

    def idle( self ):
        for i in range( self.mFloatLatency ):
            if self.mPipeline[i]:
                return False
        return True

class Memory_Datapath:
    mMemLatency : int
    mMemory : bytearray

    mNxt : typing.Tuple[ Instruction, Warp ] | None
    mPipeline : typing.List[
        typing.Tuple[ Instruction, Warp ] | None
    ]

    def __init__(
        self,
        aMemLatency : int,
        aMemory : bytearray
    ):
        self.mMemLatency = aMemLatency
        self.mMemory = aMemory
        self.mNxt = None
        self.mPipeline = self.mMemLatency * [ None ]

    def clock( self ):
        if self.mPipeline[-1]:
            instruction,warp = self.mPipeline[-1]

            for item in warp.mWorkItems:
                match instruction.mOp:
                    case ( Instruction.Op.STG ):
                        addr = item.read_reg( instruction.mDst )
                        val = instruction.mSrc[0].get_val( item, warp.mConstMem )
                        #print(f"STG: Writing {val} to {addr}")
                        self.mMemory[addr:addr+4] = struct.pack( 'I', val )
                    case ( Instruction.Op.LDG ):
                        addr = instruction.mSrc[0].get_val( item, warp.mConstMem )
                        val = struct.unpack( 'I', self.mMemory[addr:addr+4] )[0]
                        #print(f"LDG: Read {val} from {addr}")
                        item.write_reg( instruction.mDst, val )
                    case _:
                        assert 0

        for i in range( self.mMemLatency - 1 ):
            self.mPipeline[ self.mMemLatency - i - 1 ] = self.mPipeline[ self.mMemLatency - i - 2 ]

        self.mPipeline[0] = self.mNxt

        self.mNxt = None

    def print( self, prefix='' ):
        str = "|"
        for i in range( self.mMemLatency ):
            if self.mPipeline[i]:
                instr,warp = self.mPipeline[i]
                str += f"{warp.get_color()}{instr.mOp.name:6}{colors.DEF}|"
            else:
                str += f"{6*' '}|"
        print(f"{prefix} MEM  {str}")

    def is_hazard( self, aWarp : Warp, aHazRegs : typing.List[ int ] ) -> bool:
        for i in range( self.mMemLatency - 1 ):
            if self.mPipeline[i] is None:
                continue
            instr,warp = self.mPipeline[i]

            if warp != aWarp:
                continue

            # There can only be a hazard on a load, not a store
            if instr.mOp == Instruction.Op.LDG:
                # Check for hazards between instructions in the pipeline
                # and registers associated with the instruction to be added
                for reg in aHazRegs:
                    if reg == instr.mDst:
                        return True
        return False

    def push( self, aInstruction : Instruction, aWarp : Warp ):
        self.mNxt = (aInstruction, aWarp)

    def idle( self ):
        for i in range( self.mMemLatency ):
            if self.mPipeline[i]:
                return False
        return True

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
        aMemLatency : int,
        aMemory : bytearray
    ):
        self.mWarpWidth = aWarpWidth
        self.mWarpInterleave = aWarpInterleave
        self.mIntLatency = aIntLatency
        self.mFloatLatency = aFloatLatency

        self.mIntDatapath = Int_Datapath( aIntLatency=aIntLatency )
        self.mFloatDatapath = Float_Datapath( aFloatLatency=aFloatLatency )
        self.mMemDatapath = Memory_Datapath( aMemLatency=aMemLatency, aMemory=aMemory )

        self.mActiveWarps = []
        self.mWarpIdx = 0
        self.mCidx = 0

    def at_capacity( self ) -> bool:
        return len( self.mActiveWarps ) == self.mWarpInterleave

    def insert_warp( self, aWarp : Warp ):
        assert len( self.mActiveWarps ) < self.mWarpInterleave
        aWarp.set_color( clist[self.mCidx] )
        self.mCidx = ( self.mCidx + 1 ) % len( clist )
        self.mActiveWarps.append( aWarp )

    def clock( self ):
        for i in range( len( self.mActiveWarps ) ):
            self.mWarpIdx = ( self.mWarpIdx + 1 ) % len( self.mActiveWarps )

            warp = self.mActiveWarps[ ( i + self.mWarpIdx ) % len( self.mActiveWarps ) ]
            instr = warp.peek_instr()

            if instr == None:
                continue

            if instr.mDatapath:
                if self.is_hazard( instr, warp ):
                    continue
                match instr.mDatapath:
                    case Instruction.Datapath.INT:
                        self.mIntDatapath.push( instr, warp )
                    case Instruction.Datapath.FLT:
                        self.mFloatDatapath.push( instr, warp )
                    case Instruction.Datapath.MEM:
                        self.mMemDatapath.push( instr, warp )
                    case _:
                        assert 0
            else:
                match instr.mOp:
                    case Instruction.Op.NOP:
                        pass
                    case Instruction.Op.EXIT:
                        self.mActiveWarps.remove( warp )
                    case _:
                        assert 0

            warp.pop_instr()
            break

        self.mIntDatapath.clock()
        self.mFloatDatapath.clock()
        self.mMemDatapath.clock()

    def is_hazard( self, aInstruction : Instruction, aWarp : Warp ) -> bool:
        haz_regs = []
        match aInstruction.mDatapath:
            case Instruction.Datapath.INT | Instruction.Datapath.FLT:
                for src in aInstruction.mSrc:
                    if type( src ) == SrcOperand:
                        if src.mTp == SrcOperand.Tp.REG:
                            haz_regs.append( src.mVal )
            case Instruction.Datapath.MEM:
                match aInstruction.mOp:
                    case Instruction.Op.LDG:
                        # For loads, the registers with potential hazard only
                        # are the source and the source reg + 1 since memory
                        # operands are 64 bits wide
                        haz_regs.append( aInstruction.mSrc[0].mVal )
                        haz_regs.append( aInstruction.mSrc[0].mVal + 1 )
                    case Instruction.Op.STG:
                        # For stores, the registers with potential hazard
                        # are the dst and the dst reg + 1 since memory
                        # operands are 64 bits wide as well as the stored
                        # value
                        haz_regs.append( aInstruction.mDst )
                        haz_regs.append( aInstruction.mDst + 1 )
                        haz_regs.append( aInstruction.mSrc[0].mVal )
        return (
            self.mIntDatapath.is_hazard( aWarp=aWarp, aHazRegs=haz_regs ) or
            self.mFloatDatapath.is_hazard( aWarp=aWarp, aHazRegs=haz_regs ) or
            self.mMemDatapath.is_hazard( aWarp=aWarp, aHazRegs=haz_regs )
        )

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
    
    def idle( self ) -> bool:
        if len( self.mActiveWarps ):
            return False
        return (
            self.mIntDatapath.idle() and
            self.mFloatDatapath.idle() and
            self.mMemDatapath.idle()
        )

class SM:
    mNumSchedulers : int
    mSchedulers : typing.List[ Scheduler ]
    mMemorySize : int
    mMemory : bytearray
    mReadyWarps : typing.List[ Warp ]
    mWarpWidth : int
    mNumReg : int
    mIntLatency : int
    mFloatLatency : int
    mMemLatency : int

    def __init__(
        self,
        aNumSchedulers : int,
        aMemorySize : int,
        aWarpWidth : int,
        aWarpInterleave : int,
        aIntLatency : int,
        aFloatLatency : int,
        aMemLatency : int,
        aNumReg : int
    ):
        self.mMemorySize = aMemorySize
        self.mMemory = bytearray( self.mMemorySize )
        self.mNumSchedulers = aNumSchedulers
        self.mSchedulers = []
        for i in range( self.mNumSchedulers ):
            self.mSchedulers.append(
                Scheduler(
                    aWarpWidth=aWarpWidth,
                    aWarpInterleave=aWarpInterleave,
                    aIntLatency=aIntLatency,
                    aFloatLatency=aFloatLatency,
                    aMemLatency=aMemLatency,
                    aMemory=self.mMemory
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
        aConstMem : bytearray,
        aArgs : typing.List[ np.ndarray ]
    ):
        # Group Size must be a multiple of warp width
        assert aGroupSize % self.mWarpWidth == 0
        num_warp = aGroupSize // self.mWarpWidth

        assert aGlobalSize % aGroupSize == 0
        num_group = aGlobalSize // aGroupSize

        mem_idx = 0
        cmem_idx = 0x140
        for arg in aArgs:
            bb = arg.tobytes()
            self.mMemory[mem_idx:mem_idx+len(bb)] = bb
            aConstMem[cmem_idx:cmem_idx+8] = struct.pack( 'Q', mem_idx )
            mem_idx += len(bb)
            cmem_idx += 8

        aConstMem[0x8:0xc] = struct.pack( 'I', aGroupSize )

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
                if len( self.mReadyWarps ) != 0:
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

    def idle( self ) -> bool:
        for scheduler in self.mSchedulers:
            if not scheduler.idle():
                return False
        return len( self.mReadyWarps ) == 0

sm = SM(
    aNumSchedulers=4,
    aMemorySize=1024*1024,
    aWarpWidth=32,
    aWarpInterleave=4,
    aIntLatency=3,
    aFloatLatency=3,
    aMemLatency=3,
    aNumReg=16
)

print(f"{CLS}")
sm.print()

instr = parse_instructions('foo.sasm','_Z6KernelPfS_')

cmem = bytearray( 1024 )
sm.exec_kernel(
    128,
    128,
    aInstructions=instr,
    aConstMem=cmem,
    aArgs=[
        np.arange( 128 ).astype( np.float32 ),
        np.arange( 128 ).astype( np.float32 )
    ]
)

hexdump( sm.mMemory[0:1024] )

try:
    while not sm.idle():
        #time.sleep(1)
        input("type anything")

        #print(f"{CLS}")
        sm.clock()

        sm.print()
except KeyboardInterrupt:
    pass
finally:
    hexdump( sm.mMemory[0:1024] )



res =  np.frombuffer( sm.mMemory[0:128*4], dtype=np.float32 )
print(res[-1])
plt.plot(res)
plt.show()

exit()










