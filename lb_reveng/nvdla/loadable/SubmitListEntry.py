# automatically generated by the FlatBuffers compiler, do not modify

# namespace: loadable

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SubmitListEntry(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SubmitListEntry()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSubmitListEntry(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # SubmitListEntry
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SubmitListEntry
    def Id(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    # SubmitListEntry
    def TaskId(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # SubmitListEntry
    def TaskIdAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint16Flags, o)
        return 0

    # SubmitListEntry
    def TaskIdLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubmitListEntry
    def TaskIdIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def SubmitListEntryStart(builder):
    builder.StartObject(2)

def Start(builder):
    SubmitListEntryStart(builder)

def SubmitListEntryAddId(builder, id):
    builder.PrependUint16Slot(0, id, 0)

def AddId(builder, id):
    SubmitListEntryAddId(builder, id)

def SubmitListEntryAddTaskId(builder, taskId):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(taskId), 0)

def AddTaskId(builder, taskId):
    SubmitListEntryAddTaskId(builder, taskId)

def SubmitListEntryStartTaskIdVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def StartTaskIdVector(builder, numElems):
    return SubmitListEntryStartTaskIdVector(builder, numElems)

def SubmitListEntryEnd(builder):
    return builder.EndObject()

def End(builder):
    return SubmitListEntryEnd(builder)
