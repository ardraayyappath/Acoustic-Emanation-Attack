import numpy as np
import soundfile as sf
from scipy.fft import fft
import matplotlib.pyplot as plt 
import math
import string
from sklearn.neural_network import MLPClassifier
from numpy import genfromtxt
alphabets = string.ascii_lowercase
def extractKeyStroke (fileName, maxClicks, threshold):

    # This function computes the fft over a 'moving window', for each window
    # the value of the bins are summed and if they are above a certain
    # threshold it means that we have a key stroke
    #
    # When I meet an index j for which the binSums(j) is above the threshold
    # then I extract the sound knowing in advance the length which is 4410
    #
    # Remark: of course the threshold is strongly dependent on the window size

    # Here I set some variables according to training or sample mode

    arr, freq = sf.read(fileName)
    arr = arr.T

    # dropout first 550 sample points   
    rawSound = arr[549:,]

    # I define the size of the window meaning the range of values in which
    # I will compute the FFT
    winSize = 15
    winNum = math.floor(len(rawSound)/winSize)
    clickSize = 44100*0.08  # 44100 Hz * 0.08 seconds

    # Here I will place in position j-th the sum of all the bins
    # for the j-th window
    binSums = np.zeros(winNum)

    for i in range(0, winNum):
        currentWindow = np.fft.fft(rawSound[winSize*i : winSize*(i+1)])
        for j in range(0, len(currentWindow)):
            binSums[i] = binSums[i]+np.absolute(currentWindow[j])
    
    # If I keep the window small I will have more accurate results
    # since the range in which the noise is summed up is smaller
    # Of course I obtain multiple times values that are above the threshold
    # so I need to consider just the first one for every interval corresponding
    # to a key stroke length
    # A key stroke or click lasts for 0.1 seconds approximately
    # The sampling is at 44100 so there are 4410 values in a Click
    #
    # binSums(i) is the sum of the bins within the i-th windows
    # clickPositions(j) contains the beginning index of the j-th click
    # When do binSums(i) and binSums(i+k) belong to different clicks?
    # when the difference k*winSize > 4410

    clickPositions = np.zeros(maxClicks)
    j = 0
    h = 0
    offsetToNextClick = math.ceil(clickSize/winSize)
    while ((h<len(binSums)) and j < maxClicks):
        if (binSums[h] > threshold):
            clickPositions[j] = (h+1)*winSize 
            j = j+1

            # I just need the first index corresponding to the click start
            # so I adjust 'i' to avoid considering the other binSums within
            # the click duration
            h = h+offsetToNextClick
        else:
            h = h+1

    # Let's see how many individual clicks were recognized
    k = 0
    clickRecognized = 0

    while (k<len(clickPositions)):
        if(clickPositions[k] != 0):
            clickRecognized = clickRecognized+1
        k = k+1

    # Here I actually extract the key strokes
    numOfClicks = clickRecognized
    keys = np.zeros([numOfClicks,int(clickSize)])

    for i in range(0, numOfClicks):
        if (clickPositions[i] != 0):
            startIndex = clickPositions[i]-101  # -100 otherwise I get only the hit peak without touch peak
            endIndex = startIndex+clickSize-1   # -1 to have exactly 4410 values
            if ((startIndex > 0) and (endIndex < len(rawSound))):
                keys[i,:] = rawSound[int(startIndex):int(endIndex)+1]

    # Now, from the whole key stroke I just want the push peak which last 10ms
    # hence there are 441 values in it
    push_peak_size = 441
    pushPeak = np.zeros([numOfClicks,push_peak_size])

    for i in range(0, numOfClicks):
        pushPeak[i,:] = keys[i, 0:push_peak_size] 

    return pushPeak,clickRecognized, keys

def main():
    input = np.zeros((2600, 441))
    target = np.zeros(2600)
    secret_input = np.zeros((3*8,441))
    test = np.zeros((26*8,441))
    test_target = np.zeros(26*8)
    threshold = 17
     
    clicks = 0
    alpha = 'a'
    test_list = []
    for i in range(0,26):
        test_target[i*8:(i*8)+8] = i
    
    for i in range(0, 26): 
        test_list.append(alpha) 
        alpha = chr(ord(alpha) + 1)  
    

# 100 input from each file, and 26 input files --> target space is 2600

    for i in range(2600):
        target[i] = np.floor(i/100)
    
    for j, i in enumerate(alphabets):
        clicks = 0
        s = "data/"  + i + ".wav"
        k = 0
        while clicks < 99:
           
            print("Threshold test %d" %(threshold - k))
            peak,clicks,keys = extractKeyStroke(s, 100, threshold-k)
            k += 1
        # The peak from the extractKeyStroke function is noted for training
        input[j*100:j*100+len(peak), :] = abs(fft(peak))
        print(input[j*100:j*100+100])
        print("round : %d clicks %d" %(j , clicks))
    np.savetxt("fft.csv", input, delimiter=",") 

    for i in range(2600):
        target[i] = np.floor(i/100)
    input = genfromtxt('fft.csv', delimiter=',')
    
    mlp = MLPClassifier(hidden_layer_sizes=(120), verbose=False, max_iter=100000, tol=1e-4)
    mlp.fit(input, target)
    print("Overall training accuracy")
    print(mlp.score(input, target))
  
#  Prerprocessed data for test audio files has been stored inside the test.csv file to save time and use in predict.py
    for j,i in enumerate(alphabets):
        s = "data/" + i + "-test.wav"
         
        k = 0
        clicks = 0
        while clicks < 8:
             
            new_data,clicks,keys = extractKeyStroke(s, 8, threshold-k)
            k += 1
        
        s_data = abs(fft(new_data))
        test[j*8:j*8+len(new_data), :] = s_data
    np.savetxt("test.csv", test, delimiter=",")
    new_data = np.zeros([8,441])
    test = genfromtxt('test.csv', delimiter=',')
    print("Overall test accuracy")
    print(mlp.score(test, test_target))
    print("")
    for j in range(26):
        print(test_list[j], end  = "")
        print("-test.wav prediction")
        s_data = test[j*8:j*8+len(new_data), :]
        a = mlp.predict(s_data)
        print(a)
        out = a.astype(np.int)
        count = 0
        for i in out:
            if(i != j):
                count += 1
             
            print(test_list[i], end = " ") 
        print("")
        print("Accuracy : ",(abs((count-8))/8))
        print("")
    

# Prerprocessed data from secret audio files has been stored inside the secret.csv file to save time

    for i in range(3):
        s = "data/secret" + str(i) + ".wav"
        print(s)
        k = 0
        clicks = 0
        while clicks < 8:
            #print("Threshold test for test set %d" %(threshold - k))
            new_data,clicks,keys = extractKeyStroke(s, 8, threshold-k)
            k += 1
        #print("clicks %d" %(clicks))
        s_data = abs(fft(new_data))
        test[i*8:i*8+len(new_data), :] = s_data
    np.savetxt("secret.csv", test, delimiter=",")
    secret_input = genfromtxt('secret.csv', delimiter=',')
    for i in range(3):
        s_data = secret_input[i*8:i*8+len(new_data), :]
        a = mlp.predict(s_data)
        #uncomment to see predicted results interms of labels
        print(a)
        out = a.astype(np.int)
        print("secret key ",i+1)
        for i in out:
            #predicted output 
            print(test_list[i], end = " ") 
        print("")

if __name__ == "__main__":
    main()