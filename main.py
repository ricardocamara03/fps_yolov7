#!/usr/bin/python

# No desktop: conda activate yolov7_custom

import time
import signal
import threading
import sys
import cv2
import csv

import iv_VideoCapture as iv_cap
import yolov7_detector as y7detector

SHOW_CAMERA = False
MEASURE_FPS = True

FPS_SAMPLES = 120
fps_list = []


if __name__ == '__main__':
    print("Initializing main...")

    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) < 2 and MEASURE_FPS:
        print("ERROR! Pass the name of the csv output file!!!")
        exit()

    video_capture = iv_cap.iv_VideoCapture(0, 720, 480, True, 0.0001).start()
    cone_detector = y7detector.yolov7_detector("./models/yolov7-tiny-cone-02.pt")

    start_time = time.time()
    counter    = 0

    while True:
        try:
            ret, frame, time_stamp = video_capture.read()
            detections = cone_detector.yolo_detect(frame)
            # print("Nº detections:", len(detections))

            if SHOW_CAMERA:
                for i in detections:
                    cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), (255,0,0), 2)

                cv2.imshow('webcam', frame)
                if cv2.waitKey(1) == 27 : # press scape button to end the program
                    break

            if MEASURE_FPS:
                counter += 1
                if (time.time() - start_time) > 1:
                    freq = counter / (time.time() - start_time)
                    fps_list.append([freq])
                    print("FPS: ", freq, ", len(fps_list):", len(fps_list))
                    counter    = 0
                    start_time = time.time()

        except KeyboardInterruput:
            break

        if MEASURE_FPS and len(fps_list) >= FPS_SAMPLES:
            break

    if MEASURE_FPS:
        print("Saving FPS... ")
        fields = ['fps'] 
        with open(str(sys.argv[1]) + '.csv', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(fps_list)
        print("Done!")

    video_capture.stop()

    if SHOW_CAMERA:
        cv2.destroyAllWindows()