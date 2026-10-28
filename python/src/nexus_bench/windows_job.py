"""Own a suspended Windows child and its descendants until the benchmark finishes.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import ctypes
from ctypes import wintypes as w


def adopt_and_resume(process):
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    class Basic(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',w.DWORD),
                  ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',w.DWORD),
                  ('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[('counters',ctypes.c_ulonglong*6)]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                  ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    class Thread(ctypes.Structure):
        _fields_=[('size',w.DWORD),('usage',w.DWORD),('id',w.DWORD),('owner',w.DWORD),
                  ('base',w.LONG),('delta',w.LONG),('flags',w.DWORD)]
    definitions = {
        'CreateJobObjectW':([ctypes.c_void_p,w.LPCWSTR],w.HANDLE),
        'SetInformationJobObject':([w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD],w.BOOL),
        'AssignProcessToJobObject':([w.HANDLE,w.HANDLE],w.BOOL),
        'CreateToolhelp32Snapshot':([w.DWORD,w.DWORD],w.HANDLE),
        'Thread32First':([w.HANDLE,ctypes.POINTER(Thread)],w.BOOL),
        'Thread32Next':([w.HANDLE,ctypes.POINTER(Thread)],w.BOOL),
        'OpenThread':([w.DWORD,w.BOOL,w.DWORD],w.HANDLE),
        'ResumeThread':([w.HANDLE],w.DWORD),
        'CloseHandle':([w.HANDLE],w.BOOL)}
    for name,(arguments,result) in definitions.items():
        fn=getattr(kernel,name); fn.argtypes=arguments; fn.restype=result
    job=kernel.CreateJobObjectW(None,None)
    if not job: raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits=Extended(); limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job,int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        snapshot=kernel.CreateToolhelp32Snapshot(4,0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
        try:
            thread=Thread(); thread.size=ctypes.sizeof(thread)
            found=False
            more=kernel.Thread32First(snapshot,ctypes.byref(thread))
            while more:
                if thread.owner == process.pid:
                    handle=kernel.OpenThread(2,False,thread.id)  # THREAD_SUSPEND_RESUME
                    if not handle: raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if kernel.ResumeThread(handle)==0xffffffff: raise ctypes.WinError(ctypes.get_last_error())
                    finally: kernel.CloseHandle(handle)
                    found=True; break
                more=kernel.Thread32Next(snapshot,ctypes.byref(thread))
            if not found: raise RuntimeError('suspended benchmark primary thread not found')
        finally: kernel.CloseHandle(snapshot)
    except BaseException:
        kernel.CloseHandle(job)
        raise
    return lambda: kernel.CloseHandle(job)
