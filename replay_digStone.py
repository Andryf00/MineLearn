import cv2
import pickle

replay4=pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\15000_2_99.0.pkl', "rb"))#SHOW THIS AS CHOP TREE

cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
for i in range(3450,10000):
    cv2.imshow("Input", replay4[i])
    
    cv2.waitKey(50)