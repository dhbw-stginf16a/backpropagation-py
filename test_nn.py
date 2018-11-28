# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:45:11 2018

@author: dirk reichardt
"""
import numpy
import backpropagation as net


NN = net.feedForwardNetwork()
 
correctClassifications = 0
lastCorrect = 0
last_error = 1000

learned=False
number = 5
numberOfTestcases = 5
iterations = 0
old = 0
  
# The network is configured with 2 input neurons, 5 hidden
# neurons and 4 output neurons (one for each class).
  
inputDim  = 2
hiddenDim = 5
outputDim = 4
  
NN.configure(inputDim,hiddenDim,outputDim)
NN.init()
NN.setEpsilon(0.0001)
NN.setLearningRate(0.3)
  
print ("Generate training dataset:\n")

trainIn = numpy.zeros((5,2))
teach = numpy.zeros((5,4))

trainIn[0][0] = 6/15
trainIn[0][1] = 9/15
teach[0][0] = 0
teach[0][1] = 1
teach[0][2] = 0
teach[0][3] = 0    

trainIn[1][0] = 13/15
trainIn[1][1] = 12/15
teach[1][0] = 1
teach[1][1] = 0
teach[1][2] = 0
teach[1][3] = 0
    
trainIn[2][0] = 4/15
trainIn[2][1] = 4/15
teach[2][0] = 0
teach[2][1] = 0
teach[2][2] = 0
teach[2][3] = 1
    
trainIn[3][0] = 7/15
trainIn[3][1] = 5/15
teach[3][0] = 0
teach[3][1] = 0
teach[3][2] = 1
teach[3][3] = 0
        
trainIn[4][0] = 14/15
trainIn[4][1] = 15/15
teach[4][0] = 1
teach[4][1] = 0
teach[4][2] = 0
teach[4][3] = 0
    
# note: input converted to [0,1] range (neuron netinput)
  
for i in range(0,number):
    print("["+str(i)+"] "+ str(trainIn[i][0]*15) + " "+ str(trainIn[i][1]*15) + " -> ("+ str(teach[i][0])+" "+str(teach[i][1])+" "+str(teach[i][2])+" "+str(teach[i][3])+")")
    
  
    
print("\nStarting:\n")

while (correctClassifications < number):
    
    o=numpy.zeros(outputDim)
    t=numpy.zeros(outputDim)
    
    for i in range(0,number):
        iterations=iterations+1
        
        for j in range(0,inputDim):
            NN.setInput(j,trainIn[i][j])
      
        learned = False
        
        single_learn_iterations = 0

        while (not learned):
            single_learn_iterations = single_learn_iterations+1
            NN.apply()

            for j in range(0,outputDim):
                o[j] = NN.getOutput(j)
                t[j] = teach[i][j]

            error = NN.energy(t,o,outputDim);

            if (error > NN.getEpsilon()):
                NN.backpropagate(t)
                                 
            else:
                learned = True
        
        print("backpropagations: "+str(single_learn_iterations))

    # get status of learning

            
    correctClassifications = 0
    total_error = 0
    for i in range(0,number):
        for j in range(0,inputDim):
            NN.setInput(j,trainIn[i][j])
                    
        NN.apply()

        for j in range(0,outputDim):
            o[j] = NN.getOutput(j)
            t[j] = teach[i][j]
        
        error = NN.energy(t,o,outputDim)
        total_error += error

        if (error < NN.getEpsilon()):
            correctClassifications=correctClassifications+1
      
        
    # total error
    last_error = total_error

    #if (not (lastCorrect == correctClassifications)):
    print("[" + str(iterations/number) +"] >> Korrekte: " + str(correctClassifications) +" Fehler : "+ str(total_error))
     
    lastCorrect = correctClassifications

print("Iterationen:"+ str(iterations/number) + "\n")

  
