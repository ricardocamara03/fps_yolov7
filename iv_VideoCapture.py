import os
import cv2
import time
import numpy
from   threading import Thread, Lock, Event
from   datetime  import datetime

class iv_VideoCapture:
    """iv_VideoCapture Constructor"""
    def __init__(self, src = 0, width = 320, height = 240, sequential = False, sleep_time = 0.0):
        self.src     = src
        self.width   = width
        self.height  = height
        self.stream  = None
        self.grabbed = False

        #Try to initiate video source
        try:
            if os.path.exists(src):
                self.stream = cv2.VideoCapture(src)
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except:
            self.stream.release()
            print ("WebcamVideoStream: Could not get video source." )

        if not self.stream == None:
            (self.grabbed, self.frame) = self.stream.read()
        else:
            self.frame     = numpy.ones((self.height, self.width, 3), numpy.uint8)
            self.frame[:]  = (120, 0, 0)

        self.started    = False
        self.read_lock  = Lock()
        self.time_start = time.time()

        self.event_grab = Event()
        self.event_read = Event()
        self.sequential = sequential
        self.sleep_time = sleep_time
        self.event_read.set()

    def start(self) :
        if self.started :
            print ("WebcamVideoStream: Capturing already started.")
            return None

        self.started = True
        self.thread  = Thread(target=self.update, args=())
        self.thread.start()

        return self

    def update(self) :
        while self.started:
            #Testing synchronization...
            #Wait to get new data
            #======================
            time.sleep(self.sleep_time)
            if self.sequential:
                self.event_read.wait()
            #======================

            try:
                (grabbed, frame) = self.stream.read()
            except:
                grabbed = False

            #Testing synchronization...
            #Clearing to get new data
            #======================
            if self.sequential:
                self.event_read.clear()
            #======================

            self.read_lock.acquire()

            if not self.stream == None:
                self.grabbed, self.frame = grabbed, frame

            if not self.grabbed :
                if not self.stream == None:
                    self.stream.release()

                self.frame     = numpy.ones((self.height, self.width, 3), numpy.uint8)
                self.frame[:]  = (120, 0, 0)

                if time.time() - self.time_start > 5 :
                    try:
                        if os.path.exists(self.src):
                            self.stream = cv2.VideoCapture(self.src)
                            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                    except:
                        self.stream.release()
                        print ("WebcamVideoStream: Could not get video source." )

                    self.time_start = time.time()

            #Testing synchronization...
            #======================
            self.event_grab.set()
            #====================== 

            self.read_lock.release()

    def read(self) :
        #Testing synchronization...
        #======================
        self.event_grab.wait()
        #======================

        self.read_lock.acquire()

        if self.frame is None:
            self.frame     = numpy.ones((self.height, self.width, 3), numpy.uint8)
            self.frame[:]  = (120, 0, 0)

        frame = self.frame.copy()
        ret   = self.grabbed

        #Testing synchronization...
        #===================
        self.event_grab.clear()
        if self.sequential:
            self.event_read.set()
        #===================        

        self.read_lock.release()

        return ret, frame, datetime.timestamp(datetime.now())

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()
