#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# geo_coords_ex1.py
#
# Simple Example for SparkFun ublox GPS products 
#------------------------------------------------------------------------
#
# Written by  SparkFun Electronics, July 2020
# 
# Do you like this library? Help support SparkFun. Buy a board!
# https://sparkfun.com
#==================================================================================
# GNU GPL License 3.0
# Copyright (c) 2020 SparkFun Electronics
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#==================================================================================
# Example 1
# This example sets up the serial port and then passes it to the UbloxGPs
# library. From here we call geo_coords() and to get longitude and latitude. I've
# also included heading of motion here as well. 

import serial
import datetime

from ublox_gps import UbloxGps

port = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
gps = UbloxGps(port)

def run():

    try:
        print("Listening for UBX Messages")
        with open("gps_data.txt", "a") as file: # 파일을 추가 모드('a')로 열기
            while True:
                try:
                    geo = gps.geo_coords()
                    longitude = geo.lon
                    latitude = geo.lat
                    heading = geo.headMot
                    now = datetime.datetime.now()

                    # 결과를 콘솔에 출력
                    print('*'*50)
                    # print(time.strftime('%c', now))
                    print(now)
                    # print(geo)
                    print("Longitude: ", geo.lon)
                    print("Latitude: ", geo.lat)
                    print("Heading of Motion: ", geo.headMot)
                    print('*'*50)

                    # 결과를 파일에 저장
                    file.write(f"{now} Longitude: {longitude}, Latitude: {latitude} , Heading: {heading}\n")

                except (ValueError, IOError) as err:
                    print(err)

    finally:
        port.close()


if __name__ == '__main__':
    run()
