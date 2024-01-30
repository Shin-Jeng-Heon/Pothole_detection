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


# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="point", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=255.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")
parser.add_argument("--camera", type=str, default="/dev/video0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

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

# the mask image is half the size
half_width = int(opt.width/2)
half_height = int(opt.height/2)

# allocate the output images for the overlay & mask
img_overlay = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
img_mask = jetson.utils.cudaAllocMapped(half_width * half_height * 4 * ctypes.sizeof(ctypes.c_float))

# create the camera and display
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()
count = 0
pothole_count =1
temp = False
# process frames until user exits

# csvfile = open("/mnt/e7edbf37-e345-4c57-9b53-d075f037a001/jetson-inference/build/aarch64/bin/data.csv", "w")
# csvwriter = csv.writer(csvfile)

#time
cnt=0
t1=time.time()
while display.IsOpen():
		now = time.localtime()

		img, width, height = camera.CaptureRGBA()
		
		#img = jetson.utils.cudaToNumpy(img_overlay, width, height, 4 )
		
	
		#img = jetson.utils.cudaFromNumpy(img)
		

		# process the segmentation network
		net.Process(img, width, height, opt.ignore_class)

		# generate the overlay and mask
		net.Overlay(img_overlay, width, height, opt.filter_mode)
		#net.Mask(img_mask, width/2, height/2, opt.filter_mode)


		img_2 = jetson.utils.cudaToNumpy(img_overlay, width, height, 4 )
		#ROI Set
		if(temp==False):
			img2 = cv2.rectangle(img_2, (700,500), (330,300), (0, 255, 0), 3)
		else:
			img2 = cv2.rectangle(img_2, (700,500), (330,300), (255, 0, 0), 3)
		temp = False
		img_3 = cv2.cvtColor(img_2,cv2.COLOR_RGBA2BGR)

		#ROI
		#img_3 = cv2.rectangle(img_3 , (690,500), (330,300), (0, 255, 0), 3)

		#for i in range (370,490,20):
			#for j in range(382,787,20):


		for i in range (300,500,45):
			for j in range(330,850,45):
				a,s,d,f=img_2[i,j]
				if a>=120 :
					tempS=int(s)
					tempD=int(d)
					if (s-tempS)!=0 and (d-tempD)!=0:
						temp = True
						#[pothole count, location, frmae, time]
						# csvwriter.writerow([str(pothole_count),"location",str(i),str(j),"frame",str(count),
						# "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)])
						#consol print
						print("{:2}".format(str(pothole_count)) + " detection " + "(" + str(i)+", "+ str(j) + ") frame : " + str(count) )
						#print("  ",a,s,d,f)
						cv2.imwrite("./test_image/test"+str(count)+".jpg", img_3)
						pothole_count = pothole_count +1
					break

			# '''
			# for i in range (300,500,30):
			# 	for j in range(330,690,30):
			# 	a,s,d,f=img_2[i,j]
			# 	print("  ",a,s,d,f)
			# 	if a>=251:
			# 		temp = True
			# 		#print("road damage detection!")
			#
			# 		#cv2.imwrite("./test_image/test"+str(count)+".jpg", img_3)
			# 		break
			# '''




			# render the images

			

			#img_overlay = jetson.utils.cudaFromNumpy(frame_rgba)




			display.BeginRender()
			display.Render(img_overlay, width, height)
			#display.Render(img_mask, width/2, height/2, width)
			display.EndRender()

			count = count + 1
			#print(count)
			# update the title bar
			display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

		# display.BeginRender()
		# display.Render(img_overlay, width, height)
		# # display.Render(img_mask, width/2, height/2, width)
		# display.EndRender()
		#
		# count = count + 1
		# # print(count)
		# # update the title bar
		# # display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
		cnt+=1
		if(cnt>=1000):
			break
t1=time.time()-t1
print("fps: ",cnt/t1)
print("cnt: ",cnt)
print("time",t1)
# csvfile.close()
