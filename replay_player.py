import cv2
import pickle
import matplotlib.pyplot as plt
import minerl
import numpy as np




replay= pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\7000_1.pkl', "rb"))
replay2= pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\15000_2_99.0.pkl', "rb"))
replay_syr=pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\syr7997_8.0.pkl', "rb"))
#replay3= pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\10000_4.pkl', "rb"))

replay4=pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\15000_2_99.0.pkl', "rb"))#SHOW THIS AS CHOP TREE

replay_sqil=pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\syr7499_3.0.pkl', "rb"))#cool moment around 2400




cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
#cv2.namedWindow("Input2", flags=cv2.WINDOW_NORMAL)
for i in range(3450,10000):
    #digging stone at 5k
    cv2.imshow("Input", replay[i])
    
    #cv2.imshow("Input", replay_sqil[i]) #at 2400
    cv2.imshow("Input", replay4[i])
    
    cv2.waitKey(50)


