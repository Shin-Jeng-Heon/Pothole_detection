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
import argparse
import ctypes
import sys

import copy
import numpy as np

import csv
import time
from datetime import date
from datetime import datetime

import os
import shutil
from jetson_utils import videoSource, videoOutput, Log, cudaAllocMapped, cudaConvertColor, cudaDeviceSynchronize, cudaToNumpy, cudaFromNumpy
import jetson_inference

#excel
from ExcelExport import Saver

#gps
import serial
import datetime

from ublox_gps import UbloxGps

# gps
# port = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
# gps = UbloxGps(port)


# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.",
								 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson_inference.segNet.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-cityscapes-1024x512", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="point", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=99.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")
parser.add_argument("--camera", type=str, default="/dev/video0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

# parser.add_argument("--width", type=int, default=1024, help="desired width of camera stream (default is 1280 pixels)")
# parser.add_argument("--height", type=int, default=512, help="desired height of camera stream (default is 720 pixels)")


try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson_inference.segNet(opt.network, sys.argv)


# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# the mask image is half the size
half_width = int(opt.width/2)
half_height = int(opt.height/2)

# allocate the output images for the overlay & mask
img_overlay = cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
img_mask = cudaAllocMapped(half_width * half_height * 4 * ctypes.sizeof(ctypes.c_float))

#img_overlay = jetson_utils.cudaAllocMapped(width=opt.width, height=opt.height, format='rgb32f')
#img_mask = jetson_utils.cudaAllocMapped(width=half_width, height=half_height, format='rgb32f')

# create the camera and display
#camera = jetson_utils.gstCamera(opt.width, opt.height, opt.camera)
camera = videoSource('/dev/video0')

#display = jetson_utils.glDisplay()
display = videoOutput('display://0', argv=sys.argv)

count = 0
pothole_count =1
temp = False
# process frames until user exits

# csvfile = open("/mnt/e7edbf37-e345-4c57-9b53-d075f037a001/jetson-inference/build/aarch64/bin/data.csv", "w")
# csvwriter = csv.writer(csvfile)

# delete & make dir
if (os.path.isdir("./log") == True):
	shutil.rmtree("./log")
if (os.path.isdir("./log") == False):
	os.mkdir("./log")

if (os.path.isdir("./test_image") == True):
	shutil.rmtree("./test_image")
if (os.path.isdir("./test_image") == False):
	os.mkdir("./test_image")
	
if (os.path.isdir("./full_frame_detect") == True):
	shutil.rmtree("./full_frame_detect")
if (os.path.isdir("./full_frame_detect") == False):
	os.mkdir("./full_frame_detect")

if (os.path.isdir("./full_frame") == True):
	shutil.rmtree("./full_frame")
if (os.path.isdir("./full_frame") == False):
	os.mkdir("./full_frame")
#time

import cv2

while True:
	now = time.localtime()
	cimg = camera.Capture()
	if cimg is None:
		print('img error')
		continue

	bgr_img = cudaAllocMapped(width=cimg.width, height=cimg.height, format='rgba32f')

	cudaConvertColor(cimg, bgr_img)

	#print('BGR image: ')
	#print(bgr_img)

	# make sure the GPU is done work before we convert to cv2
	cudaDeviceSynchronize()

	# convert to cv2 image (cv2 images are numpy arrays)
	cv_img = cudaToNumpy(bgr_img)

	frame = cv2.resize(cv_img, dsize=(1024, 512), interpolation=cv2.INTER_AREA)
	# frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	# rgba32f match
	# frame_rgba = frame_rgba.astype(np. float32)
	frame_rgba = frame.astype(np.float32)
	width = frame.shape[1]
	height = frame.shape[0]

	# ROI Set
	if (temp == False):
		img = cv2.rectangle(frame_rgba, (954, 480), (70, 280), (0, 255, 0), 3)
	else:
		img = cv2.rectangle(frame_rgba, (954, 480), (70, 280), (255, 0, 0), 3)
	temp = False
	img = cudaFromNumpy(frame_rgba)

	# process the segmentation network
	net.Process(img, width, height, opt.ignore_class)

	# generate the overlay and mask
	net.Overlay(img_overlay, width, height, opt.filter_mode)
	net.Mask(img_mask, int(width / 2), int(height / 2), opt.filter_mode)

	# print(img_overlay)
	img_2 = cudaToNumpy(img_overlay, width, height, 4)
	img_3 = cv2.cvtColor(img_2, cv2.COLOR_RGBA2BGR)
	img_4 = cv2.cvtColor(img_2, cv2.COLOR_RGBA2RGB)

	#print(width, height)
	omask = cudaToNumpy(img_mask, int(width / 2), int(height / 2), 4)
	img_mask2 = cv2.cvtColor(omask, cv2.COLOR_RGBA2BGR)

	# mask copy
	cmask = img_mask2.copy()
	# mask : bgr -> grayscale
	mask_gray = cv2.cvtColor(cmask, cv2.COLOR_BGR2GRAY)
	# binary : less than 100 -> 0
	ret, mask_thres = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
	# cv2.imshow("d", img_3[0:50,0:50])

	mask_thres = mask_thres.astype(np.uint8)

	# find coutour's vertex
	contours, hierarchy = cv2.findContours(mask_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# draw vertex, green
	cv2.drawContours(mask_thres, contours, -1, (0, 255, 0), 4)
	# draw vertex, circle, blue

	for i in contours:
		for j in i:
			cv2.circle(mask_thres, tuple(j[0]), 1, (255, 0, 0), -1)

	# find center
	# c0 = contours[0]count
	for c in contours:
		M = cv2.moments(c)
		# print(M,type(M))

		if (M['m00'] > 0):
			cx = int(M["m10"] / M['m00'])
			cy = int(M['m01'] / M['m00'])
			# print("found something : (" + str(cx) + ", " + str(cy) + ")")
			cv2.rectangle(mask_thres, (325, 240), (100, 140), (255, 0, 0), 3)
			cv2.circle(mask_thres, (cx, cy), 1, (0, 0, 0), -1)

			if (cx <= 325 and cx >= 100 and cy <= 240 and cy >= 140):
				# print(mask_thres.shape)
				cv2.imwrite("./log/test" + str(count) + ".jpg", mask_thres)
				# mb, mg, mr = img_mask2[cx, cy]
				mb, mg, mr = img_mask2[cy, cx]
				value = max(mb, mg, mr)
				if value > 0 and value == mr: # class 별 색상 찾기

					# GPS
					'''
					geo = gps.geo_coords()
					longitude = geo.lon
					latitude = geo.lat
					heading = geo.headMot
					now = datetime.datetime.now()
					'''

					print("{:2}".format(str(pothole_count)) + " detection " + "(" + str(cx * 2) + ", " + str(
						cy * 2) + ") frame : " + str(count))
					img_name = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
					print(img_name)

					# Excel 데이터 추가
					# Saver.add_data(img_name, longitude, latitude)

					cv2.imwrite("./test_image/" + img_name + ".jpg", img_3)

					pothole_count += 1
					break

	# render the images

	# bgr img
	# output = cudaFromNumpy(img_3)

	output = cudaFromNumpy(img_4)
	new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	cv2.imwrite("./full_frame/"+str(count)+".jpg", new_frame)
	cv2.imwrite("./full_frame_detect/"+str(count)+".jpg", img_3)

	# print('--frame_rgba: ',type(frame_rgba))
	# print('--img_overay: ',type(img_overlay))
	# print('--img_3: ',type(img_3))
	#	print('*'*50)
	#	print(img_3.shape,img_overlay.shape)


	#display.BeginRender()
	#print(type(img_overlay))
	display.Render(output)
	#display.Render(img_mask, width/2, height/2, width)
	#display.EndRender()

	count = count + 1
	if not camera.IsStreaming() or not display.IsStreaming():
		break

# Excel 파일 저장
# Saver.save_to_excel()

#print(count)
# update the title bar
#display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

# csvfile.close()
