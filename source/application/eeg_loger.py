# python version >= 2.5
import sys,os
import ctypes
import threading
import time
from ctypes import cdll
from ctypes import CDLL
from ctypes import c_int
from ctypes import c_uint
from ctypes import pointer
from ctypes import c_char_p
from ctypes import c_float
from ctypes import c_double
from ctypes import byref

"""
This file is used to record EEG signal using the Emotiv EPOC device. Recording is running in separated thread so it can
stopped by user. To run recording this files need to be in this folder as well: edk.dll and edk_utils.dll. They are
part of Emotiv SDK.
"""

try :
    if sys.platform.startswith('win32'):     
        libEDK = cdll.LoadLibrary("edk.dll")
    if sys.platform.startswith('linux'):
        srcDir = os.getcwd()    
        libPath = srcDir + "/libedk.so.1.0.0"        
        libEDK = CDLL(libPath)
except :
    print ('Error : cannot load dll lib')

ED_COUNTER = 0
ED_INTERPOLATED=1
ED_RAW_CQ=2
ED_AF3=3
ED_F7=4
ED_F3=5
ED_FC5=6
ED_T7=7
ED_P7=8
ED_O1=9
ED_O2=10
ED_P8=11
ED_T8=12
ED_FC6=13
ED_F4=14
ED_F8=15
ED_AF4=16
ED_GYROX=17
ED_GYROY=18
ED_TIMESTAMP=19
ED_ES_TIMESTAMP=20
ED_FUNC_ID=21
ED_FUNC_VALUE=22
ED_MARKER=23
ED_SYNC_SIGNAL=24

targetChannelList = [ED_COUNTER,ED_AF3, ED_F7, ED_F3, ED_FC5, ED_T7,ED_P7, ED_O1, ED_O2, ED_P8, ED_T8,ED_FC6, ED_F4, ED_F8, ED_AF4, ED_GYROX, ED_GYROY, ED_TIMESTAMP, ED_FUNC_ID, ED_FUNC_VALUE, ED_MARKER, ED_SYNC_SIGNAL]
header = ['COUNTER','AF3','F7','F3', 'FC5', 'T7', 'P7', 'O1', 'O2','P8', 'T8', 'FC6', 'F4','F8', 'AF4','GYROX', 'GYROY', 'TIMESTAMP','FUNC_ID', 'FUNC_VALUE', 'MARKER', 'SYNC_SIGNAL']

is_recording = 0


def start_recording(filename):
    class StartThread (threading.Thread):
        def __init__(self, filename):
            threading.Thread.__init__(self)
            self.filename = filename
            print(self.filename)
        def run(self):
            write = sys.stdout.write
            eEvent      = libEDK.EE_EmoEngineEventCreate()
            eState      = libEDK.EE_EmoStateCreate()
            userID            = c_uint(0)
            nSamples   = c_uint(0)
            nSam       = c_uint(0)
            nSamplesTaken  = pointer(nSamples)
            #da = zeros(128,double)
            data     = pointer(c_double(0))
            user                    = pointer(userID)
            composerPort          = c_uint(1726)
            secs      = c_float(1)
            datarate    = c_uint(0)
            readytocollect    = False
            option      = c_int(0)
            state     = c_int(0)


            option = 1

            if option == 1:
                connect_param = c_char_p(b'Emotiv Systems-5')
                if libEDK.EE_EngineConnect(connect_param) != 0:
                    print("Emotiv Engine start up failed.")
            elif option == 2:
                connect_param = c_char_p(b'127.0.0.1')
                if libEDK.EE_EngineRemoteConnect(connect_param, composerPort) != 0:
                    print("Cannot connect to EmoComposer on")
            else :
                print("option = ?")

            print("Start receiving EEG Data! Press any key to stop logging...\n")
            f = open(self.filename, 'w')
            print(header, file=f)

            hData = libEDK.EE_DataCreate()
            libEDK.EE_DataSetBufferSizeInSec(secs)

            print("Buffer size in secs:", secs)
            while is_recording:
                try:
                    state = libEDK.EE_EngineGetNextEvent(eEvent)
                    if state == 0:
                        eventType = libEDK.EE_EmoEngineEventGetType(eEvent)
                        libEDK.EE_EmoEngineEventGetUserId(eEvent, user)
                        if eventType == 16: #libEDK.EE_Event_enum.EE_UserAdded:
                            print("User added")
                            libEDK.EE_DataAcquisitionEnable(userID,True)
                            readytocollect = True

                    if readytocollect==True:
                        libEDK.EE_DataUpdateHandle(0, hData)
                        libEDK.EE_DataGetNumberOfSample(hData,nSamplesTaken)
                        # print("Updated :", nSamplesTaken[0])
                        if nSamplesTaken[0] != 0:
                            nSam=nSamplesTaken[0]
                            arr=(ctypes.c_double*nSamplesTaken[0])()
                            ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
                            #libEDK.EE_DataGet(hData, 3,byref(arr), nSam)
                            #data = array('d')#zeros(nSamplesTaken[0],double)
                            for sampleIdx in range(nSamplesTaken[0]):
                                for i in range(22):
                                    libEDK.EE_DataGet(hData,targetChannelList[i],byref(arr), nSam)
                                    # print(arr[sampleIdx],",", end=' ', file=f)
                                # print('\n', file=f)
                    time.sleep(0.2)
                    # print(hData)
                except KeyboardInterrupt:
                    break
            libEDK.EE_DataFree(hData)

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------
            libEDK.EE_EngineDisconnect()
            libEDK.EE_EmoStateFree(eState)
            libEDK.EE_EmoEngineEventFree(eEvent)

            print("Ending main")

    start_thread = StartThread(filename)
    start_thread.start()

