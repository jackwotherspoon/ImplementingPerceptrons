#dependencies as well as matplotlib and sklearn
import pandas as pd
import numpy as np
import random

outFile=False #set to true when we want to write to output file

#reading data using pandas (note that original data was slightly modified to include bias column of 1's
columns=['bias','area','perimeter','compactness','length','width','asymmetry coeffecient','length of kernel groove','class']
train=pd.read_csv('trainSeeds.csv', names=columns)
test=pd.read_csv('testSeeds.csv', names=columns)

#sorting data into inputs and outputs (X and Y) for training data
X_train=train.drop('class',axis=1)
Y_train=train['class']
tool_train=X_train.drop("bias",axis=1)     #remove bias column for tool since sklearn auto does bias

#sorting data into inputs and outputs (X and Y) for testing data
X_test=test.drop('class',axis=1)
Y_test=test['class']
tool_test=X_test.drop("bias",axis=1)     #remove bias column for tool since sklearn auto does bias
testSize=np.size(X_test,0) #number of rows in testing data (45)

#get number of objects (rows) and attributes (columns)
obj=np.size(X_train,0)      #number of training objects
attr=np.size(X_train,1)     #number of training attributes (columns)

#set learning rate and number of iterations for training
c=0.001                 #learning rate for training set
iterations=1000         #number of iterations through each row

#create weights
weight_Ros=[]
for i in range(attr):
    weight_Ros.append(random.uniform(-1, 1)) #initial weights chosen randomly for bias and inputs


#wouldnt allow me to use same random weights as input to both training sets so i had to make a second one
weight_Can=[]
for i in range(attr):
    weight_Can.append(random.uniform(-1, 1))

#add initial weight to output file
if outFile:
    f=open('Assignment1_Output.txt','w')
    f.write("\n\nInitial Weights:\n")
    f.write("Rosa Perceptron Initial Weights: "+str((weight_Ros)))
    f.write("\nCanadian Perceptron Initial Weights: "+str((weight_Can)))

#extract values from data
x=X_train.values
y=Y_train.values

xTest=X_test.values
y_test=Y_test.tolist()

#setup answer strings that hold a 0 or 1 depending on the class of the wheat that should be correctly identified, easier than class
y_Rosa=[]
y_Canadian=[]
for i in range(0,55):
    y_Rosa.append(0)
    y_Canadian.append(0)
for i in range(0,55):
    y_Rosa.append(1)
    y_Canadian.append(0)
for i in range(0,55):
    y_Rosa.append(0)
    y_Canadian.append(1)
#print(y_Rosa)
#print(y_Canadian)

#activation function predicts whether perceptron should display a 0 or 1
def predict(row,weights):
    summ=0
    for i in range(0,attr):
        summ+= weights[i] * row[i]      #activation threshold
    return 1.0 if summ>=0.0 else 0.0

#training function that takes in the data and random weights and returns updated weights based on simple feedback learning
def training(data,weight, iterations,learningRate,answer):
    for epoch in range(iterations):    #loop until number of iterations has been reached through each row
        correct=0
        for i in range(len(data)):    #loop through each row of data and make prediction
            row=data[i]
            prediction=predict(row,weight)     #make prediction based on current weights
            error=answer[i]-(prediction)       #calculate error which is (actual-predicted)
            if error==0:
                correct+=1
            for j in range(len(weight)):
                weight[j]=weight[j]+(learningRate*row[j]*error)    #update weight accordingly with respect to error, only needed one eqn here instead of two since i used 0 and 1 as my outputs which can only lead to 1 or -1 for error
        acc=correct/len(data)
        #print(acc)
    return weight

#train the two perceptrons and return the trained weights for each wheat
weights_Rosa=(training(x,weight_Ros,iterations,c,y_Rosa))
weights_Canadian=(training(x,weight_Can,iterations,c,y_Canadian))

#add final weights to output file
if outFile:
    f.write("\n\nFinal Weights:\n")
    f.write("Rosa Perceptron Final Weights: "+str((weights_Rosa)))
    f.write("\nCanadian Perceptron Final Weights: "+str((weights_Canadian)))

#testing fuction which runs trained weights through testing data to predict their wheat types
def testing(data,weight,answer):
    guess=[]
    for i in range(testSize):               #loop through each row of testing data
        row=data[i]
        guess.append(predict(row,weight))   #add each answer to list of answers
    #print("Predicted: "+str(guess)+" Actual: "+str(answer[i]))
    return guess

testing(xTest,weights_Rosa,y_test)
testing(xTest,weights_Canadian,y_test)

#results function combines data for both perceptrons thus allowing a truth table to be built to classify all 3 wheat types
def results(data,weight1,weight2,answers):
    results_Rosa=(testing(data,weight1,answers))             #output of testing data for Rosa perceptron
    results_Canadian=testing(data,weight2,answers)           #output of testing data for Canadian perceptron
    result=[]
    for i in range(testSize):
        if results_Rosa[i]==0 and results_Canadian[i]==0:    #if output of Rosa is 0 and Canadian is 0 then by induction the wheat must be Kama
          result.append(1)
        elif results_Rosa[i]==1 and results_Canadian[i]==0:  #if output of Rosa is 1 and Canadian is 0 then wheat is Rosa
            result.append(2)
        elif results_Rosa[i]==0 and results_Canadian[i]==1:  #if output of Rosa is 0 and Canadian is 1 then wheat is Canadian
            result.append(3)
        else:
            result.append(4)                                 #if both perceptrons classify the wheat as a 1 then we have a false positive and label it 4
    return result
final_result=results(xTest,weights_Rosa,weights_Canadian,y_test)       #get final result of tested data
print("Actual Labels from Test Data: "+str(y_test))
print("Predicted Labels of Test Data: "+str(final_result))
print("A label of 4 means that both Rosa and Canadian outputed positively.")

#accuracy function to decide how accurate the trained perceptrons classified the test data
def accuracy(tested,actual):
    correct=0
    for i in range(len(tested)):
        if tested[i]==actual[i]:
            correct+=1
            accRate=correct/len(tested)
    return accRate
print("Accuracy on test data:"+str(accuracy(final_result,y_test)))
if outFile:
    f.write("\n\nTraining:")
    f.write("\nTotal number of iterations for training was set to 1000.\n")
    f.write("This was the terminating criteria as I figured 1000 was enough to get great accuracy without overfitting.\n")
    f.write("Could have also used an accuracy threshold as well as terminating criteria to stop from overfitting.\n")
    f.write("Accuracy for training of both Rosa and Canadian perceptron was 0.9878 after 1000 iterations with learning rate of 0.001.\n")
    f.write("\n\nTest set results: (4 classifies that both perceptrons classified it positively")
    f.write("\nActual Classes from Given Data    :  "+str(y_test))
    f.write("\nPredicted Classes from Perceptrons:  "+str(final_result))
    f.write("\n\nAccuracy on Test Set: "+str(accuracy(final_result,y_test)))

#Using the sklearn tool!!!!
print("\n\nUsing tool!")
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
perceptron=Perceptron(max_iter=1000,tol=0.1)
perceptron.fit(tool_train,Y_train)
y_pred=perceptron.predict(tool_test)
print("Train accuracy: %.2f \n" % perceptron.score(tool_train,Y_train))
print("Test accuracy: %.2f \n" % perceptron.score(tool_test,Y_test))

confusion_mat=confusion_matrix(Y_test.values,y_pred)
print("Confusion Matrix\n")
print(confusion_mat)
print("Precision and recall from tool: \n")
class_report=classification_report(Y_test.values,y_pred)
print(class_report)
if outFile:
    f.write("\n\nUsing the sklearn Tool!!!\n")
    f.write("Training accuracy: %.2f" % perceptron.score(tool_train,Y_train))
    f.write("\nTesting accuracy: %.2f" % perceptron.score(tool_test,Y_test))
    f.write("\nConfusion Matrix for sklearn Tool: \n")
    f.write(np.array2string(confusion_mat))
    f.write("\nPrecision and Recall: \n")
    f.write(class_report)
    f.close