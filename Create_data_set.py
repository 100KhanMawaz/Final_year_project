import os

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2
import os
import pickle
cap = cv2.VideoCapture(0)

hands = mp.solutions.hands
hands_mesh=hands.Hands(static_image_mode=True,min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

DATA_DIR='./data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux  = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path)) #imread function takes url of the image and reads it as an image and loads it into img variable
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#for proccessing we need RGB images so this function helps us converting image to RGB image

        results = hands_mesh.process(img_rgb)#now whatever properties we defined for hand_mesh like to detect hand and detection confidence etc we want to apply all those properties to img_rgb so this is how we do it
        #now the results is an object that has too many member functions here we will use multi_hand_landmarks which gives us a list of 21 landmark points in hand

        if results.multi_hand_landmarks:
            #if we do have any hand in the screen then only we are going to get landmarks otherwise results.multi_hand_landmarks will have null so make sure to hava condition before performing any operations
            for i in results.multi_hand_landmarks:
                #iterating over all 21 landmarks
                draw.draw_landmarks(img,i,hands.HAND_CONNECTIONS)#this function helps draw landmarks on image which is provided in 1st paramter and we know we need co-ordinates to plot landmarks so 2nd paramter we are passing the co-ordinates, basically here i is nothing but the x,y,z co-ordinates for every given landmarks so this draw_landmarks function will run 21 times and all 21 times we are providing different x,y,z
                # x=i.landmark.x # we cannot access like this because at each ith iteration i is having 21 landmarks with same name like see
                # landmark:{
                # x:0.423421
                # y:0.342134
                # z:0.324124
                # }
                # landmark:{
                # x:
                # y:
                # z:
                # }
                #landmark:{
                # x:
                # y:
                # z:
                # }
                # and so on till 21 landmark so if we try doing like i.landmark.x the compiler is confused that your asking x co-ordinate of which landmark out of 21 so to manage that we can run a loop 21 times and access all the landmakr's co-ordinates like i.landmark[i].x
                for itr in range(21):
                   x = i.landmark[itr].x
                   y = i.landmark[itr].y
                   data_aux.append(x)
                   data_aux.append(y)

            data.append(data_aux) #whenever we are done with all the 21 landmarks of each image then put it in data and along with it jis directory mein abhi iterate hora hai like 0,1,2.etc to usko labels mein daal denge basically ek directory mein 100 iterations hoga and data ke andar 100 image ka data_aux dalenge and har baar dir_ same rahega qki ek folder mein 100 samples hai
            labels.append(dir_)

        # plt.figure()
        # plt.imshow(img)

#pickle and export the model
f = open('data.pickle','wb')
pickle.dump({'data':data, 'labels':labels}, f)
f.close()