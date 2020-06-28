import numpy as np
import torch

# array = np.array([[False  ,True ,False ,False ,False ,False ,False ,False ,False ,False],
#                   [False , True ,False ,False ,False ,False ,False ,False ,False ,False],
#                   [False, False ,False ,False ,False  ,True ,False ,False ,False ,False]])



# def threshold_tensor_batch(pred, base=0.50):
#     p_max = torch.max(pred, dim=1)[0]
#     pivot = torch.cuda.FloatTensor([base]).expand_as(p_max)
#     threshold = torch.min(p_max, pivot) 
#     pred_label = torch.ge(pred, threshold.unsqueeze(dim=1))
#     return pred_label

def weight_cal(arr,per = 0.3):

    val = arr.shape[0] * per
 #   result = [0,0,0,0,0,0,0,0,0,0]
    result = np.sum(arr,axis = 0)
    for i in range(10):
        #print(result[i])
        result[i] = int(result[i] > val)
    return result


import csv
# label = csv.reader(open('Label.csv','r'))
pro = csv.reader(open("Prob.csv",'r'))
# for row in label:
#     print(row)

# key = row[0]
key = 'test126'
pred = []

coumt = 0 
with open("test_pred.csv","w") as csvfile: 
    writer = csv.writer(csvfile)


    for row in pro:
        coumt +=1 
        print(row[0],row[1])

        if key != row[0]:
            pred =np.array(pred)
            pred = np.sum(pred,axis = 0)
            pred = pred/np.max(pred)
            # np.set_printoptions(precision=4)
            #pred = np.around(pred, decimals=4, out=None)
            pred = (pred > 0.5)
            print(pred)
          
            string = ''
#             flag = 0
            for i in range(10):
                if pred[i] == 1:
                    string += str(i)
                    string += ";"


            writer.writerow([key,string[:-1]])
            # input()
        
            pred = []
            key = row[0]
    
        row[1]= row[1].split(';')
        for i in range(10):
            row[1][i] = float(row[1][i])
        pred.append(row[1])


    print(key)
    print(pred)
    pred = np.sum(pred,axis = 0)
    pred = pred/np.max(pred)
    pred = np.around(pred, decimals=4, out=None)
 
            # np.set_printoptions(precision=4)
            #pred = np.around(pred, decimals=4, out=None)
    pred = (pred > 0.5)
    print(pred)
    string = ''
#             flag = 0
    for i in range(10):
        if pred[i] == 1:
            string += str(i)
            string += ";"


    writer.writerow([key,string[:-1]])

