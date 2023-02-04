from PSOClustering import PSO_Clustering
import pandas as pd
import numpy as np
import math



def runSequence(data, popNum, MaxIt, numOfSequence, sequenceSteps):
    for i in range(numOfSequence):
        bestK_list = []
        bestCost_list =[]
        for j in range(sequenceSteps):
            print("------sequence num: ", i,'----- sequence step: ', j )
            if (j == 0):
                k = np.random.randint(2, math.sqrt(len(data)))
                bestKSofar = k
                bestK_list.append(bestKSofar)
                cost, pos = PSO_Clustering(data, k, popNum, MaxIt, True)
                bestCostSofar = cost
                bestCost_list.append(bestCostSofar)
                print("cost: ", cost)
                print("pos: ", pos)
            else:
                k = k + np.random.randint(-10, 10)
                if (k < 2):
                    k = 2
                cost, pos = PSO_Clustering(data, k, popNum, MaxIt, True)
                if (cost < bestCostSofar):
                    bestCostSofar = cost
                    bestKSofar = k
                bestK_list.append(bestKSofar)
                bestCost_list.append(bestCostSofar)
        print("best k list: ", bestK_list)
        print("best cost list: ", bestCost_list)
        iman_breakPoint = 13


if __name__ == '__main__':
    sequenceNum = 3
    sequenceSteps = 5
    sequence_PSOPup = 5
    sequence_MaxIt = 10
    dataset = pd.read_csv('Iman_Data', names = ["f1", "f2", "f3", "targets"])
    target = dataset["targets"]
    trainDataset = dataset.drop(columns=["targets"])
    bestK = runSequence(trainDataset, sequence_PSOPup, sequence_MaxIt, sequenceNum, sequenceSteps)



