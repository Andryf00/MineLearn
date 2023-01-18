import cv2
import pickle
import matplotlib.pyplot as plt
import minerl
"""if __name__=='__main__':
    cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
    # Sample some data from the dataset!
    data = minerl.data.make("MineRLObtainDiamond-v0")
    trajs = data.get_trajectory_names()
    for n, traj in enumerate(trajs):
        for j, sample in enumerate(data.load_data(traj, include_metadata=True)):
            print(sample)
            cv2.imshow("Input", sample[0]['pov'])
            print(sample[0]['inventory'])
            print("action;",sample[1])
            #cv2.imshow("Input2", replay2[i])
            cv2.waitKey()
exit()"""


d = {
    "11":5,
    "16":1,
    "19":3,
    "35":4,
    "67":2,
    "99":4
}


"""
data=[0.073, 0.062, 0.065, 0.056, 0.048, 0.045, 0.040, 0.036, 0.037, 0.032, 0.029, 0.028, 0.026, 0.023, 0.020, 0.021, 0.023, 0.020, 0.0225, 0.027, 0.021, 0.020, 0.019, 0.021, 0.020]


# create the plot
plt.plot(data)

# add a title
plt.title("Training loss DigStone Agent")

# add a legend
plt.legend(["trainig loss"])

# show the plot
plt.show()
"""
replay= pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\7000_1.pkl', "rb"))
replay2= pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\15000_2_99.0.pkl', "rb"))
replay_syr=pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\syr7997_8.0.pkl', "rb"))
#replay3= pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\10000_4.pkl', "rb"))

replay_sqil=pickle.load(open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\syr7499_3.0.pkl', "rb"))#cool moment around 2400




cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
#cv2.namedWindow("Input2", flags=cv2.WINDOW_NORMAL)
for i in range(0,7550):
    #digging stone at 5k
    cv2.imshow("Input", replay[i])
    
    #cv2.imshow("Input", replay_sqil[i])
    cv2.waitKey(50)


