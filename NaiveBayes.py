import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import clean
import firePCA

def main():
    y, x = clean.clean()

    newY, newX = data_classification(y,x)

    accuracies = list()
    totalaccuracy = 0.0

    #80% training 20% testing
    foldSize = int(x.shape[0]/5)
    for i in range(0,5):
        xtest = newX[(i*foldSize):((i+1)*foldSize),:]
        ytest = newY[(i*foldSize):((i+1)*foldSize)]
        xtrain = np.concatenate((newX[:(i*foldSize),:],newX[((i+1)*foldSize):,:]))
        ytrain = np.concatenate((newY[:(i*foldSize)],newY[((i+1)*foldSize):]))       
        
        priorProbabilities = priors_prob(ytrain)

        yassigned = assignment(xtest,xtrain,ytrain,priorProbabilities)

        accuracy = analysis(ytest,yassigned)
        accuracies.append(accuracy)

        print("Round %d Accuracy: %f" % (i+1, accuracy))

        totalaccuracy += accuracy

    totalaccuracy /= 5
    print("\nOverall Accuracy: %f" % (totalaccuracy))

    
    plot1 = plt.scatter(np.arange(0,ytest.shape[0]),ytest[:,0] == yassigned[:,0], c = 'blue', marker='x')
    plt.xlabel('Testing Points')
    plt.ylabel('Class')
    plt.title('Classification Actual vs. Predicted For States 2019 & 2020')

    plt.show()

    plot2 = plt.scatter(np.arange(0,5),accuracies, color = 'red', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy For Each Round')
    plt.show()

    x = firePCA.firePCA()
    newY, newX = data_classification(y,x)
    accuracies = list()
    totalaccuracy = 0.0

    for i in range(0,5):
        xtest = newX[(i*foldSize):((i+1)*foldSize),:]
        ytest = newY[(i*foldSize):((i+1)*foldSize)]
        xtrain = np.concatenate((newX[:(i*foldSize),:],newX[((i+1)*foldSize):,:]))
        ytrain = np.concatenate((newY[:(i*foldSize)],newY[((i+1)*foldSize):]))       
        
        priorProbabilities = priors_prob(ytrain)

        yassigned = assignment(xtest,xtrain,ytrain,priorProbabilities)

        accuracy = analysis(ytest,yassigned)
        accuracies.append(accuracy)

        print("Round %d Accuracy: %f" % (i+1, accuracy))

        totalaccuracy += accuracy

    totalaccuracy /= 5
    print("\nOverall Accuracy: %f" % (totalaccuracy))

    plot3 = plt.scatter(np.arange(0,ytest.shape[0]),ytest[:,0] == yassigned[:,0], c = 'blue', marker='x')
    plt.xlabel('Testing Points')
    plt.ylabel('Class')
    plt.title('Classification Actual vs. Predicted For States 2019 & 2020 after PCA')

    plt.show()

    plot4 = plt.scatter(np.arange(0,5),accuracies, color = 'red', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy For Each Round after PCA')
    plt.show()




def data_classification(yData, xData):
    #create cuts for each of the features to classify all features
    new_x = np.empty(xData.shape)
    for i in range(0,xData.shape[1]):
        upperX = max(xData[:,i])
        lowerX = min(xData[:,i])
        span = upperX - lowerX
        cut1 = (span/3) + lowerX
        cut2 = (2 * span/3) + lowerX
        lowIndexes = xData[:,i] < cut1
        new_x[lowIndexes,i] = 0
        medIndexes = np.logical_and(xData[:,i] >= cut1,xData[:,i] < cut2)
        new_x[medIndexes,i] = 1
        highIndexes = xData[:,i] >= cut2
        new_x[highIndexes,i] = 2

    

    #create cuts for y
    new_y = np.empty(yData.shape)

    upperY = max(yData[:,0])
    cut1 = upperY/3
    cut2 = 2*upperY/3

    #classify all the y points
    lowIndexes = yData < cut1
    new_y[lowIndexes] = 0

    medIndexes = np.logical_and(yData >= cut1,yData < cut2)
    new_y[medIndexes] = 1

    highIndexes = yData >= cut2
    new_y[highIndexes] = 2

    new_y = new_y.astype(int)

    return new_y, new_x


def priors_prob(yData):
    yLow = np.sum(yData == 0)
    yMed = np.sum(yData == 1)
    yHigh = np.sum(yData == 2)

    total = yLow + yMed + yHigh

    priorProbabilities = np.array([yLow/total,yMed/total,yHigh/total])
    
    return priorProbabilities


def assignment(xtest,xtrain,ytrain,priorProbabilities):
    y_assignments = np.empty((xtest.shape[0],1)) #our assignments from model

    #different classes of x from training data set of x
    xLow = xtrain[ytrain[:,0] == 0]
    xMed = xtrain[ytrain[:,0] == 1]
    xHigh = xtrain[ytrain[:,0] == 2]

    #number of y for each class, add 1 rule in case 0 exist for certain class
    yLow = np.sum(ytrain == 0) + 1
    yMed = np.sum(ytrain == 1) + 1
    yHigh = np.sum(ytrain == 2) + 1

    #go through each point
    for i in range(0,xtest.shape[0]):
        #prior probs
        probLow = priorProbabilities[0]
        probMed = priorProbabilities[1]
        probHigh = priorProbabilities[2]

        #get likelihood based on each feature
        for j in range(0,xtest.shape[1]):
            feat = xtest[i,j]
            probLow *= (np.sum(xLow[:,j] == feat)+1)/yLow
            probMed *= (np.sum(xMed[:,j] == feat)+1)/yMed
            probHigh *= (np.sum(xHigh[:,j] == feat)+1)/yHigh

        y_assignments[i,0] = np.argmax([probLow,probMed,probHigh])

    return y_assignments


def analysis(ytest,yassigned):
    correct = sum(ytest == yassigned)
    return correct/ytest.shape[0]

if __name__ == "__main__":
    main()