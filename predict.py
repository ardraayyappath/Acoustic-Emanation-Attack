import numpy as np
import soundfile as sf
from scipy.fft import fft
import matplotlib.pyplot as plt 
import math
import string
from sklearn.neural_network import MLPClassifier
from numpy import genfromtxt

def main():
    input = np.zeros((2600, 441))
    target = np.zeros(2600)
    secret_input = np.zeros((3*8,441))
    test = np.zeros((26*8,441))
    test_target = np.zeros(26*8)
    threshold = 15
    clicks = 0
    alpha = 'a'
    test_list = []
    for i in range(0,26):
        test_target[i*8:(i*8)+8] = i
    
    for i in range(0, 26): 
        test_list.append(alpha) 
        alpha = chr(ord(alpha) + 1)  
    

    for i in range(2600):
        target[i] = np.floor(i/100)
    input = genfromtxt('fft.csv', delimiter=',')
    
    mlp = MLPClassifier(hidden_layer_sizes=(120), verbose=False, max_iter=100000, tol=1e-4)
    mlp.fit(input, target)
    #print("Overall training accuracy")
    #print(mlp.score(input, target))
  
    accuracy = []
    new_data = np.zeros([8,441])
    test = genfromtxt('test.csv', delimiter=',')
    #print("Overall test accuracy")
    #print(mlp.score(test, test_target))
    #print("Running test data sets in the model")
    #print("")
    for j in range(26):
        print(test_list[j], end  = "")
        print("-test.wav prediction")
        s_data = test[j*8:j*8+len(new_data), :]
        a = mlp.predict(s_data)
        print(a)
        out = a.astype(int)
        count = 0
        for i in out:
            if(i != j):
                count += 1
            #predicted output 
            print(test_list[i], end = " ") 
        
        print("\nAccuracy : ",(abs((count-8))/8),"\n")
        accuracy.append([test_list[j],(abs((count-8))/8)])
     
    np.savetxt("accuracy.csv", accuracy,fmt='%s', delimiter=",")
    print(accuracy)
    
    

 
    secret_input = genfromtxt('secret.csv', delimiter=',')
    for i in range(3):
        s_data = secret_input[i*8:i*8+len(new_data), :]
        a = mlp.predict(s_data)
        #uncomment to see predicted results interms of labels
        #print(a)
        out = a.astype(int)
        print("secret key ",i+1)
        for i in out:
            #predicted output 
            print(test_list[i], end = " ") 
        print("")

if __name__ == "__main__":
    main()