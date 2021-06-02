# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:51:33 2019

@author: Sayak
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread(r"D:\Datasets\standard_test_images\standard_test_images\lena_color_512.tif")

face_data=r"D:\Datasets\data\haarcascades\haarcascade_frontalface_alt.xml"
classifier=cv2.CascadeClassifier(face_data)

faces=classifier.detectMultiScale(img)

for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

cv2.imshow('Lena',img)
cv2.waitKey(0)
cv2.destroyAllWindows()