from nvdla.loadable.Loadable import Loadable
from nvdla.loadable.Version import Version
from nvdla.loadable.TaskListEntry import TaskListEntry 
from nvdla.loadable.Interface import Interface
from nvdla.loadable.MemoryListEntry import MemoryListEntry
from nvdla.loadable.MemoryDomain import MemoryDomain
from nvdla.loadable.MemoryFlags import MemoryFlags
from nvdla.loadable.AddressListEntry import AddressListEntry
from nvdla.loadable.EventListEntry import EventListEntry
from nvdla.loadable.Blob import Blob
from nvdla.loadable.TensorDescListEntry import TensorDescListEntry
from nvdla.loadable.RelocListEntry import RelocListEntry
from nvdla.loadable.SubmitListEntry import SubmitListEntry
from nvdla.loadable.EventType import EventType
from nvdla.loadable.EventOp import EventOp
from nvdla.loadable.DataFormat import DataFormat
from nvdla.loadable.DataType import DataType
from nvdla.loadable.DataCategory import DataCategory
from nvdla.loadable.PixelFormat import PixelFormat
from nvdla.loadable.PixelMapping import PixelMapping


INDENT = 3

import flatbuffers

def print_loadable(obj: Loadable, spaces:int = 0):
    """
    Print the contents of a Loadable object.
    
    version      : Version ( required );
    task_list    : [TaskListEntry];
    memory_list  : [MemoryListEntry];
    address_list : [AddressListEntry];
    event_list   : [EventListEntry];
    blobs        : [Blob];
    tensor_desc_list : [TensorDescListEntry];
    reloc_list   : [RelocListEntry];
    submit_list  : [SubmitListEntry];
    """
    #  version
    print_version(obj.Version(), 0)
    
    # task_list
    print(f'{" " * (spaces)}Task list: ({obj.TaskListLength()})')
    for i in range(obj.TaskListLength()):
        task = obj.TaskList(i)
        print_task(task, spaces + INDENT)
    
    # memory_list
    print(f'{" " * (spaces)}Memory list: ({obj.MemoryListLength()})')
    for i in range(obj.MemoryListLength()):
        mem = obj.MemoryList(i)
        print_memory(mem, spaces + INDENT)   
    
    # address_list
    print(f'{" " * (spaces)}Address list: ({obj.AddressListLength()})')
    for i in range(obj.AddressListLength()):
        addr = obj.AddressList(i)
        print_address_list(addr, spaces + INDENT)
    
    # event list
    print(f'{" " * (spaces)}Event list: ({obj.EventListLength()})')
    for i in range(obj.EventListLength()):
        event = obj.EventList(i)
        print_event_list(event, spaces + INDENT)
    
    # blobs
    print(f'{" " * (spaces)}Blob list: ({obj.BlobsLength()})')
    for i in range(obj.BlobsLength()):
        blob = obj.Blobs(i)
        print_blob(blob, spaces + INDENT)
    
    # tensor_desc_list
    print(f'{" " * (spaces)}TensorDesc list: ({obj.TensorDescListLength()})')
    for i in range(obj.TensorDescListLength()):
        tensor_desc = obj.TensorDescList(i)
        print_tensor_desc_list(tensor_desc, spaces + INDENT)
    
    # reloc_list
    print(f'{" " * (spaces)}Reloc list: ({obj.RelocListLength()})')
    for i in range(obj.RelocListLength()):
        reloc = obj.RelocList(i)
        print_reloc_list(reloc, spaces + INDENT)
    
    # submit_list
    print(f'{" " * (spaces)}Submit list: ({obj.SubmitListLength()})')
    for i in range(obj.SubmitListLength()):
        submit = obj.SubmitList(i)
        print_submit_list(submit, spaces + INDENT)
    

def print_submit_list(obj:SubmitListEntry, spaces:int = 0):
    """
    Print the contents of a SubmitListEntry object recursively.
    
    id      : ushort;
    task_id : [ushort];
    """
    task_list = [obj.TaskId(i) for i in range(obj.TaskIdLength() if obj.TaskIdLength() < 10 else 10)]
    print(f'{" " * spaces}SubmitListEntry: id: {obj.Id()}, task_id: {task_list}')
    

    
def print_reloc_list(obj:RelocListEntry, spaces:int = 0):
    """
    Print the contents of a RelocListEntry object recursively.
    
    address_id : ushort;
    write_id : ushort;
    offset : ulong;
    interface: uint;
    sub_interface: uint;
    reloc_type: ubyte;
    """
    print(f'{" " * spaces}RelocListEntry: address')
    print(f'{" " * (spaces + INDENT)}RelocListEntry address_id: {obj.AddressId()}')
    print(f'{" " * (spaces + INDENT)}RelocListEntry write_id: {obj.WriteId()}')
    print(f'{" " * (spaces + INDENT)}RelocListEntry offset: {obj.Offset()}')
    print(f'{" " * (spaces + INDENT)}RelocListEntry interface: {obj.Interface()}')
    print(f'{" " * (spaces + INDENT)}RelocListEntry sub-interface: {obj.SubInterface()}')
    print(f'{" " * (spaces + INDENT)}RelocListEntry reloc_type: {obj.RelocType()}')
    
    

def print_tensor_desc_list(obj:TensorDescListEntry, spaces:int = 0):
    """
    
    name : string;
    id : ushort;
    mem_id : ushort;
    size   : ulong;
    offset : ulong;
    data_format  : DataFormat;
    data_type    : DataType;
    data_category: DataCategory;
    pixel_format : PixelFormat;
    pixel_mapping : PixelMapping;

    n : int;
    c : int;
    h : int;
    w : int;

    stride_0 : uint;
    stride_1 : uint;
    stride_2 : uint;
    stride_3 : uint;
    stride_4 : uint;
    stride_5 : uint;
    stride_6 : uint;
    stride_7 : uint;
    """
    
    print(f'{" " * spaces}TensorDescListEntry: ')
    print(f'{" " * (spaces + INDENT)}TensorDesc ID: {obj.Id()}')
    print(f'{" " * (spaces + INDENT)}TensorDesc name: {obj.Name().decode("utf-8")}')
    print(f'{" " * (spaces + INDENT)}TensorDesc mem_id: {obj.MemId()}')
    print(f'{" " * (spaces + INDENT)}TensorDesc size: {obj.Size()}')
    print(f'{" " * (spaces + INDENT)}TensorDesc offset: {obj.Offset()}')
    
    # data_format
    data_formats = {
        DataFormat.NCHW: "NCHW",
        DataFormat.NHWC: "NHWC",
        DataFormat.UNKNOWN: "UNKNOWN",
        3: "NOT IN FBS",
    }
    print(f'{" " * (spaces + INDENT)}TensorDesc data_format: {data_formats[obj.DataFormat()]}')
    
    # data_type
    data_types = {
        DataType.UNKNOWN: "UNKNOWN",
        DataType.FLOAT: "FLOAT",
        DataType.HALF: "HALF",
        DataType.INT16: "INT16",
        DataType.INT8: "INT8",
    }
    print(f'{" " * (spaces + INDENT)}TensorDesc data_type: {data_types[obj.DataType()]}')
    
    # data_category
    data_categories = {
        DataCategory.IMAGE: "IMAGE",
        DataCategory.WEIGHT: "WEIGHT",
        DataCategory.FEATURE: "FEATURE",
        DataCategory.PLANAR: "PLANAR",
        DataCategory.BIAS: "BIAS",
    }
    print(f'{" " * (spaces + INDENT)}TensorDesc data_category: {data_categories[obj.DataCategory()]}')
    
    # pixel_format
    pixel_formats = {
        PixelFormat.R8: "R8",
        PixelFormat.R10: "R10",
        PixelFormat.R12: "R12",
        PixelFormat.R16: "R16",
        PixelFormat.R16_I: "R16_I",
        PixelFormat.R16_F: "R16_F",
        PixelFormat.A16B16G16R16: "A16B16G16R16",
        PixelFormat.X16B16G16R16: "X16B16G16R16",
        PixelFormat.A16B16G16R16_F: "A16B16G16R16_F",
        PixelFormat.A16Y16U16V16: "A16Y16U16V16",
        PixelFormat.V16U16Y16A16: "V16U16Y16A16",
        PixelFormat.A16Y16U16V16_F: "A16Y16U16V16_F",
        PixelFormat.A8B8G8R8: "A8B8G8R8",
        PixelFormat.A8R8G8B8: "A8R8G8B8",
        PixelFormat.B8G8R8A8: "B8G8R8A8",
        PixelFormat.R8G8B8A8: "R8G8B8A8",
        PixelFormat.X8B8G8R8: "X8B8G8R8",
        PixelFormat.X8R8G8B8: "X8R8G8B8",
        PixelFormat.B8G8R8X8: "B8G8R8X8",
        PixelFormat.R8G8B8X8: "R8G8B8X8",
        PixelFormat.A2B10G10R10: "A2B10G10R10",
        PixelFormat.A2R10G10B10: "A2R10G10B10",
        PixelFormat.B10G10R10A2: "B10G10R10A2",
        PixelFormat.R10G10B10A2: "R10G10B10A2",
        PixelFormat.A2Y10U10V10: "A2Y10U10V10",
        PixelFormat.V10U10Y10A2: "V10U10Y10A2",
        PixelFormat.A8Y8U8V8: "A8Y8U8V8",
        PixelFormat.V8U8Y8A8: "V8U8Y8A8",
        PixelFormat.Y8_U8V8_N444: "Y8_U8V8_N444",
        PixelFormat.Y8_V8U8_N444: "Y8_V8U8_N444",
        PixelFormat.Y10_U10V10_N444: "Y10_U10V10_N444",
        PixelFormat.Y10_V10U10_N444: "Y10_V10U10_N444",
        PixelFormat.Y12_U12V12_N444: "Y12_U12V12_N444",
        PixelFormat.Y12_V12U12_N444: "Y12_V12U12_N444",
        PixelFormat.Y16_U16V16_N444: "Y16_U16V16_N444",
        PixelFormat.Y16_V16U16_N444: "Y16_V16U16_N444",
        PixelFormat.FEATURE: "FEATURE",
        PixelFormat.FEATURE_X8: "FEATURE_X8",
    }
    print(f'{" " * (spaces + INDENT)}TensorDesc pixel_format: {pixel_formats[obj.PixelFormat()]}')
    
    # pixel_mapping
    pixel_mappings = {
        PixelMapping.INVALID_PIXEL_MAP: "INVALID_PIXEL_MAP",
        PixelMapping.PITCH_LINEAR: "PITCH_LINEAR",
    }
    print(f'{" " * (spaces + INDENT)}TensorDesc pixel_mapping: {pixel_mappings[obj.PixelMapping()]}')
    
    # n, c, h, w
    print(f'{" " * (spaces + INDENT)}n: {obj.N()}, c: {obj.C()}, h: {obj.H()}, w: {obj.W()}')
    
    # stride_0, stride_1, stride_2, stride_3, stride_4, stride_5, stride_6, stride_7
    print(f'{" " * (spaces + INDENT)}strides: {[obj.Stride0(), obj.Stride1(), obj.Stride2(), obj.Stride3(), obj.Stride4(), obj.Stride5(), obj.Stride6(), obj.Stride7()]}')


def print_blob(obj:Blob, spaces:int = 0):
    """
    Print the contents of a Blob object recursively.
    
    name        : string;
    size        : ulong;
    interface   : Interface;   // e.g.: dla1 (firmware)
    sub_interface: uint;       // e.g.: dla1-op-desc, dla1-surf-desc
    version     : Version;     // e.g.: firmware version
    data        : [ubyte];
    """
    
    print(f'{" " * spaces}Blob: ')
    print(f'{" " * (spaces + INDENT)}Blob name: {obj.Name().decode("utf-8")}')
    print(f'{" " * (spaces + INDENT)}Blob size: {obj.Size()}')
    print_interface(obj.Interface(), spaces + INDENT)
    print(f'{" " * (spaces + INDENT)}Blob sub-interface: {obj.SubInterface()}')
    print_version(obj.Version(), spaces + INDENT)
    
    # data
    data = [obj.Data(i) for i in range(obj.Size() if obj.Size() < 10 else 10)]
    print(f'{" " * (spaces + INDENT)}Blob data: ({obj.Size()}) {data}')


def print_event_list(obj:EventListEntry, spaces:int = 0):
    """
    Print the contents of a EventListEntry object recursively.
    
    id    : ushort;
    type  : EventType;
    target: ushort;
    val   : uint;
    op    : EventOp;
    """
    print(f'{" " * spaces}EventListEntry: ')
    print(f'{" " * (spaces + INDENT)}Event ID: {obj.Id()}')
    
    event_types = {
        EventType.EVENTTYPE0: "EVENTTYPE0",
        EventType.EVENTTYPE1: "EVENTTYPE1",
        EventType.EVENTTYPE2: "EVENTTYPE2",
    }
    print(f'{" " * (spaces + INDENT)}Event Type: {event_types[obj.Type()]}')
    
    print(f'{" " * (spaces + INDENT)}Event Target: {obj.Target()}')
    print(f'{" " * (spaces + INDENT)}Event Value: {obj.Val()}')
    
    event_op = {
        EventOp.SIGNAL: "SIGNAL",
        EventOp.WAIT: "WAIT",
    }
    print(f'{" " * (spaces + INDENT)}Event Op: {event_op[obj.Op()]}')
    

def print_address_list(obj:AddressListEntry, spaces:int = 0):
    """
    Print the contents of a AddressListEntry object recursively.
    
    id     : ushort;
    mem_id : ushort;
    offset : ulong;
    size   : ulong;
    """
    print(f'{" " * spaces}AddressListEntry: id: {obj.Id()}, mem_id: {obj.MemId()}, offset: {obj.Offset()}, size: {obj.Size()}')
    
    
def print_memory(obj:MemoryListEntry, spaces:int = 0):
    """
    Print the contents of a MemoryListEntry object recursively.
    
    id       : ushort;
    domain   : MemoryDomain;
    flags    : MemoryFlags;
    size     : ulong;
    alignment : uint;
    contents : [string];
    offsets : [ulong];
    bind_id : ushort;
    tensor_desc_id : ushort;
    """
    
    print(f'{" " * spaces}MemoryListEntry: ')
    print(f'{" " * (spaces + INDENT)}Memory ID: {obj.Id()}')
    print_memory_domain(obj.Domain(), spaces + INDENT)
    print_memory_flags(obj.Flags(), spaces + INDENT)
    print(f'{" " * (spaces + INDENT)}Memory size: {obj.Size()}')
    print(f'{" " * (spaces + INDENT)}Memory alignment: {obj.Alignment()}')
    
    memory_contents = [obj.Contents(i) for i in range(obj.ContentsLength() if obj.ContentsLength() < 10 else 10)]
    print(f'{" " * (spaces + INDENT)}Memory contents: ({obj.ContentsLength()}) {memory_contents}')
    
    memory_offsets = [obj.Offsets(i) for i in range(obj.OffsetsLength() if obj.OffsetsLength() < 10 else 10)]
    print(f'{" " * (spaces + INDENT)}Memory offsets: ({obj.OffsetsLength()}) {memory_offsets}')
    print(f'{" " * (spaces + INDENT)}Memory bind ID: {obj.BindId()}')
    print(f'{" " * (spaces + INDENT)}Memory tensor desc ID: {obj.TensorDescId()}')
    

def print_memory_flags(obj: MemoryFlags, spaces:int = 0):
    """
    Print the contents of a MemoryFlags object recursively.
    
    NONE = 0
    ALLOC = 1
    SET = 2
    INPUT = 4
    OUTPUT = 8
    """
    # Flags are bitwise OR'd together, so we need to check each flag
    
    flags = {
        MemoryFlags.NONE: "NONE",
        MemoryFlags.ALLOC: "ALLOC",
        MemoryFlags.SET: "SET",
        MemoryFlags.INPUT: "INPUT",
        MemoryFlags.OUTPUT: "OUTPUT",
    }
    
    print(f'{" " * spaces}MemoryFlags: {[name for bit, name in flags.items() if obj & bit]}')
    
    
def print_memory_domain(obj:MemoryDomain, spaces:int = 0):
    """
    Print the contents of a MemoryDomain object recursively.
    
    SYSTEM = 0
    SRAM = 1
    """
    
    domains = {
        MemoryDomain.SYSTEM: "SYSTEM",
        MemoryDomain.SRAM: "SRAM",
    }
    
    print(f'{" " * spaces}MemoryDomain: {domains[obj]}')

        
def print_version(obj: Version, spaces:int = 0):
    """
    Print the contents of a Version object recursively.
    """
    print(f'{" " * (spaces)}Version: ')
    print(f'{" " * (spaces + INDENT)}Major: {obj.Major()}, Minor: {obj.Minor()}, SubMinor: {obj.SubMinor()}')
    
    
def print_task(obj: TaskListEntry, spaces:int = 0):
    """
    Print the contents of a TaskListEntry object recursively.
    
    id           : ushort;
    interface    : Interface;
    instance     : short;
    address_list : [ushort];
    pre_actions  : [ushort];
    post_actions : [ushort];
    """
    
    print(f'{" " * spaces}TaskListEntry: ')
    print(f'{" " * (spaces + INDENT)}Task ID: {obj.Id()}')
    print_interface(obj.Interface(), spaces + INDENT)
    print(f'{" " * (spaces + INDENT)}Task instance: {obj.Instance()}')
    
    address_list = [obj.AddressList(i) for i in range(obj.AddressListLength() if obj.AddressListLength() < 10 else 10)]
    print(f'{" " * (spaces + INDENT)}Address List: ({obj.AddressListLength()}) {address_list}')
    
    pre_actions = [obj.PreActions(i) for i in range(obj.PreActionsLength() if obj.PreActionsLength() < 10 else 10)]
    print(f'{" " * (spaces + INDENT)}Pre Actions: ({obj.PreActionsLength()}) {pre_actions}')
    
    post_actions = [obj.PostActions(i) for i in range(obj.PostActionsLength() if obj.PostActionsLength() < 10 else 10)]
    print(f'{" " * (spaces + INDENT)}Post Actions: ({obj.PostActionsLength()}) {post_actions}')
    pass


def print_interface(obj: Interface, spaces:int = 0):
    """
    Print the contents of a Interface object recursively.
    """
    interfaces = {
        Interface.NONE: "NONE",
        Interface.DLA1: "DLA1",
        Interface.EMU1: "EMU1",
    }
    
    print(f'{" " * spaces}Interface: {interfaces[obj]}')