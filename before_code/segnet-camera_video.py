#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import cv2

import jetson.inference
import jetson.utils
import argparse
import ctypes
import sys

import copy
import numpy as np

import csv
import time
from datetime import date

import os
import shutil

#GPS module
'''
import time
import board
import busio
import adafruit_gps
import serial

uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=10)
gps = adafruit_gps.GPS(uart, debug=False)
gps.send_command(b'PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
gps.send_command(b'PMTK220,2000')
last_print = time.monotonic()
'''


# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.",
                     formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="point", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=99.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")

#parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")

parser.add_argument("--camera", type=str, default="/dev/video0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1920, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=1080, help="desired height of camera stream (default is 720 pixels)")

try:
   opt = parser.parse_known_args()[0]
except:
   print("")
   parser.print_help()
   sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

#net.FuncTest()

# allocate the output images for the overlay & mask
img_overlay = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
img_mask = jetson.utils.cudaAllocMapped(opt.width/2 * opt.height/2 * 4 * ctypes.sizeof(ctypes.c_float))
# img_mask = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
# create the camera and display
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()

cam = cv2.VideoCapture("3.mov")
# cam = cv2.VideoCapture("video/test1.mp4")

#
count = 0
pothole_count =1
temp = False
# process frames until user exits

# csvfile = open("/mnt/e7edbf37-e345-4c57-9b53-d075f037a001/jetson-inference/build/aarch64/bin/data.csv", "w")
# csvwriter = csv.writer(csvfile)


# delete & make dir
if(os.path.isdir("./test_image") == True):
	shutil.rmtree("./test_image")
if(os.path.isdir("./test_image") == False):
	os.mkdir("./test_image")

#time
while display.IsOpen():
      now = time.localtime()


      '''
      gps.update()
      # Every second print out current location details if there's a fix.
      current = time.monotonic()

      if current - last_print >= 1.0:
         last_print = current
         if not gps.has_fix: 
               # Try again if we don't have a fix yet.
               print('Waiting for fix...')
               continue
         
         # We have a fix! (gps.has_fix is true)
         # Print out details about the fix like location, date, etc.
         print('=' * 40)  # Print a separator line.
         convert_hour=gps.timestamp_utc.tm_hour+9
         print('Fix timestamp: {}/{}/{} {:02}:{:02}:{:02}'.format(
               gps.timestamp_utc.tm_mon,   # Grab parts of the time from the
               gps.timestamp_utc.tm_mday,  # struct_time object that holds
               gps.timestamp_utc.tm_year,  # the fix time.  Note you might
               convert_hour,  # not get all data like year, day,
               gps.timestamp_utc.tm_min,   # month!
               gps.timestamp_utc.tm_sec))
         print('Latitude: {0:.6f}'.format(gps.latitude))
         print('Longitude: {0:.6f}'.format(gps.longitude))
         print('Altitude: {} meters'.format(gps.altitude_m))
        '''




      #get frame
      ret, frame = cam.read()

      #convert format for video inference
      frame = cv2.resize(frame, dsize=(1024, 512), interpolation=cv2.INTER_AREA)
      frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
      # rgba32f match
      frame_rgba = frame_rgba.astype(np.float32)
      width = frame.shape[1]
      height = frame.shape[0]

      #ROI Set
      if (temp == False):
          img = cv2.rectangle(frame_rgba, (690, 500), (330, 300), (0, 255, 0), 3)
      else:
          img = cv2.rectangle(frame_rgba, (690, 500), (330, 300), (255, 0, 0), 3)
      temp = False
      img = jetson.utils.cudaFromNumpy(frame_rgba)

      # process the segmentation network
      net.Process(img, width, height, opt.ignore_class)

      # generate the overlay and mask
      net.Overlay(img_overlay, width, height, opt.filter_mode)
      net.Mask(img_mask, width/2, height/2, opt.filter_mode)

      # print(img_overlay)
      img_2 = jetson.utils.cudaToNumpy(img_overlay, width, height, 4 )
      # mask = jetson.utils.cudaToNumpy(img_mask, width, height, 4)
      img_3 = cv2.cvtColor(img_2,cv2.COLOR_RGBA2BGR)
      # mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGR)
      #ROI
      #img_3 = cv2.rectangle(img_3 , (690,500), (330,300), (0, 255, 0), 3)

      #for i in range (370,490,20):
         #for j in range(382,787,20):

      for x in range(280, 480, 20):
          for y in range(200, 650, 20):
            # a,s,d,f=img_2[j,i]
            b, g, r = img_3[x, y]
            value=max(b,g,r)
            if value>=120 and value==r :
                tempG = int(g)
                tempB = int(b)
                if (g - tempG) != 0 and (b - tempB) != 0:

                  temp = True
                  # [pothole count, location, frmae, time]
                  # csvwriter.writerow([str(pothole_count),"location",str(i),str(j),"frame",str(count),
                  # "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)])
                  # consol print
                  print("{:2}".format(str(pothole_count)) + " detection " + "(" + str(x) + ", " + str(
                      y) + ") frame : " + str(count))
                  # print("  ",a,s,d,f)
                  # print("(" + str(x)+", "+ str(y) + ")" + " : " + "( B : " + str(b) + ", G : "
                  # + str(g) +", R : " + str(r) +" )")
                  text = ('( B : ' + str(b) + ', G : ' + str(g) + ', R : ' + str(r) + ' )')
                  # img_4 = cv2.putText(img_3, text, (x, y), 1, 1, (0,0,0), 2)
                  cv2.putText(img_3, text, (y, x), 1, 1, (255, 255, 255), 2, cv2.LINE_AA)
                  # cv2.imwrite("./test_image/test"+str(count)+".jpg", img_3)
                  cv2.imwrite("./test_image/test" + str(count) + ".jpg", img_3)
                  pothole_count = pothole_count + 1
                  break

      '''
      for i in range (300,500,30):
         for j in range(330,690,30):
            a,s,d,f=img_2[i,j]
            print("  ",a,s,d,f)
            if a>=251:
               temp = True
               #print("road damage detection!")
               
               #cv2.imwrite("./test_image/test"+str(count)+".jpg", img_3)
               break
      '''




      # render the images



      #img_overlay = jetson.utils.cudaFromNumpy(frame_rgba)




      display.BeginRender()
      display.Render(img_overlay, width, height)
      display.Render(img_mask, width/2, height/2, width)
      display.EndRender()

      count = count + 1
      #print(count)
      # update the title bar
      #display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

# csvfile.close()