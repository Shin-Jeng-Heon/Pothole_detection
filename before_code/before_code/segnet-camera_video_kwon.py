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

#GPS module before
'''
import time
import board
import busio
import adafruit_gps
import serial

uart = serial.Serial("/dev/ttyACM0", baudrate=9600, timeout=10)
gps = adafruit_gps.GPS(uart, debug=False)
gps.send_command(b'PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
gps.send_command(b'PMTK220,2000')
last_print = time.monotonic()
'''

def getOpt():
      # parse the command line
      parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.",
                                       formatter_class=argparse.RawTextHelpFormatter,
                                       epilog=jetson.inference.segNet.Usage())

      parser.add_argument("--network", type=str, default="fcn-resnet18-cityscapes-1024x512",
                          help="pre-trained model to load, see below for options")
      parser.add_argument("--filter-mode", type=str, default="point", choices=["point", "linear"],
                          help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
      parser.add_argument("--ignore-class", type=str, default="void",
                          help="optional name of class to ignore in the visualization results (default: 'void')")
      parser.add_argument("--alpha", type=float, default=99.0,
                          help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 230.0)")

      # parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")

      parser.add_argument("--camera", type=str, default="/dev/video4",
                          help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
      parser.add_argument("--width", type=int, default=1920,
                          help="desired width of camera stream (default is 1280 pixels, 1920)")

      parser.add_argument("--height", type=int, default=1080,
                          help="desired height of camera stream (default is 720 pixels, 1080)")
      parser.add_argument("--videoPath", type=str, default="./test4.mp4", help="sd")

      parser.add_argument("--video", type=str, default=True, help="sd")

      try:
            opt = parser.parse_known_args()[0]
      except:
            print("")
            parser.print_help()
            sys.exit(0)
      return opt

def setModel(opt):
      # load the segmentation network

      # net.FuncTest()
      net = jetson.inference.segNet(opt.network, sys.argv)

      # set the alpha blending value
      net.SetOverlayAlpha(opt.alpha)
      return net

def setDir():
      # delete & make dir
      if (os.path.isdir("./log") == True):
            shutil.rmtree("./log")
      if (os.path.isdir("./log") == False):
            os.mkdir("./log")

      if (os.path.isdir("./test_image") == True):
            shutil.rmtree("./test_image")
      if (os.path.isdir("./test_image") == False):
            os.mkdir("./test_image")

def getFrame(cam, temp):
      # get frame
      ret, frame = cam.read()

      # convert format for video inference
      if (type(frame) == type(None)):
            pass
      else:
            frame = cv2.resize(frame, dsize=(1024, 512), interpolation=cv2.INTER_AREA)
      frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
      # rgba32f match
      frame_rgba = frame_rgba.astype(np.float32)
      width = frame.shape[1]
      height = frame.shape[0]

      # ROI Set
      if (temp == False):
            cv2.rectangle(frame_rgba, (690, 500), (330, 300), (0, 255, 0), 3)
      else:
            cv2.rectangle(frame_rgba, (690, 500), (330, 300), (255, 0, 0), 3)
            temp = False
      img = jetson.utils.cudaFromNumpy(frame_rgba)

      return img, temp, frame, width, height

def setContour(omask):
      img_mask2 = cv2.cvtColor(omask, cv2.COLOR_RGBA2BGR) # 이미지 위에 레이어(detection 결과 표시) 불러옴.
      # mask copy
      cmask = img_mask2.copy()
      # mask : bgr -> grayscale
      mask_gray = cv2.cvtColor(cmask, cv2.COLOR_BGR2GRAY)
      # binary : less than 100 -> 0
      _, mask_thres = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
      # mask_thres match
      mask_thres = mask_thres.astype(np.uint8)

      # find coutour's vertex
      contours, hierarchy = cv2.findContours(mask_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      # draw vertex, green
      cv2.drawContours(mask_thres, contours, -1, (0, 255, 0), 4)
      # draw vertex, circle, blue
      for i in contours:
            for j in i:
                  cv2.circle(mask_thres, tuple(j[0]), 1, (255, 0, 0), -1)

      return img_mask2, mask_thres, contours

def findPothole(count, mask_thres, img_mask2, img_3, frame, contours, width, height):
      pothole_count = 1
      # find center
      for c in contours:
            M = cv2.moments(c)
            # print(M,type(M))
            if (M['m00'] > 0):
                  cx = int(M["m10"] / M['m00'])
                  cy = int(M['m01'] / M['m00'])
                  # print("found something : (" + str(cx) + ", " + str(cy) + ")")
                  cv2.rectangle(mask_thres, (345, 250), (165, 150), (255, 0, 0), 3)
                  cv2.circle(mask_thres, (cx, cy), 1, (0, 0, 0), -1)

                  #if (cx <= width / 2 and cx >= 0 and cy <= height / 2 and cy >= height / 4):
                  if (165 <= cx <= 345 and 150 <= cy <= 250):
                        # print(mask_thres.shape)
                        # cv2.imwrite("./log/test" + str(count) + ".jpg", mask_thres)
                        # mb, mg, mr = img_mask2[cx, cy]
                        mb, mg, mr = img_mask2[cy, cx]
                        value = max(mb, mg, mr)
                        if value > 0 and value == mr:
                              print("{:2}".format(str(pothole_count)) + " detection " + "(" + str(cx * 2) + ", " + str(
                                    cy * 2) + ") frame : " + str(count))
                              cv2.imwrite("./log/test" + str(count) + ".jpg", frame)
                              cv2.imwrite("./test_image/test" + str(count) + ".jpg", img_3)
                              pothole_count += 1
                              break

def demo(opt):
      count = 0
      temp = False

      net = setModel(opt=opt)

      # allocate the output images for the overlay & mask
      img_overlay = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
      img_mask = jetson.utils.cudaAllocMapped(int(opt.width / 2 * opt.height / 2 * 4 * ctypes.sizeof(ctypes.c_float)))
      # img_mask = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
      # create the camera and display
      camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
      display = jetson.utils.glDisplay()

      # cam = cv2.VideoCapture("3.mov")
      cam = cv2.VideoCapture(opt.videoPath)

      # process frames until user exits

      # csvfile = open("/mnt/e7edbf37-e345-4c57-9b53-d075f037a001/jetson-inference/build/aarch64/bin/data.csv", "w")
      # csvwriter = csv.writer(csvfile)

      setDir()

      # time
      while display.IsOpen():
            # now = time.localtime()

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
            img, temp, frame, width, height = getFrame(cam=cam, temp=temp)

            # process the segmentation network
            net.Process(img, width, height, opt.ignore_class)

            # generate the overlay and mask
            net.Overlay(img_overlay, width, height, opt.filter_mode)
            net.Mask(img_mask, int(width / 2), int(height / 2), opt.filter_mode)

            img_2 = jetson.utils.cudaToNumpy(img_overlay, width, height, 4)
            img_3 = cv2.cvtColor(img_2, cv2.COLOR_RGBA2BGR)
            # img_mask2 = jetson.utils.cudaToNumpy(img_overlay, width/2, height/2, 4 )
            omask = jetson.utils.cudaToNumpy(img_mask, int(width / 2), int(height / 2), 4)

            img_mask2, mask_thres, contours = setContour(omask=omask)

            findPothole(count=count, mask_thres=mask_thres, img_mask2=img_mask2, img_3=img_3, frame=frame, contours=contours, width=width, height=height)
            
            if opt.video == True:
                display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
                display.BeginRender()
                display.Render(img_overlay, width, height)
                display.Render(img_mask, int(width / 2), int(height / 2), width)
                display.EndRender()
                count = count + 1
            else:
                pass

            # # render the images
            # # img_overlay = jetson.utils.cudaFromNumpy(frame_rgba)
            # # update the title bar
            # display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
            # display.BeginRender()
            # display.Render(img_overlay, width, height)
            # # display.Render(img_mask2, width/2, height/2, width)
            # display.Render(img_mask, width / 2, height / 2, width)
            # display.EndRender()
            # count = count + 1
            # # print("frame :", count)
            # # cv2.imwrite("./full_frame/frame" + str(count) + ".jpg", frame)

if __name__ == '__main__':
      opt = getOpt()
      demo(opt=opt)


      # csvfile.close()
