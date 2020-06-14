import csv
import os

myfile = open("newtrain.txt","w")
myfile1 = open("newtest.txt","w")

with open("DataSet/train.csv",'r') as f:
    reader = csv.reader(f)
    count = 0
    c =0
    for row in reader:
        if len(row[1]) > 1:
            count += 1
            print(row[0],row[1])
            row[1] = row[1].split(';')
            l= [0 for i in range(10)]
            for  i in row[1]:
                l[int(i)] = 1
            str1 =''
            for i in l:
                str1 += ' '
                str1 += str(i)
            path = os.path.join("DataSet/train",row[0])
            for (root, dirs, files) in os.walk(path):
                imagepath =[os.path.join(root,i) for i in files]
                fileinfo = [i+ str1 for i in imagepath]
                                #print(fileinfo[0])
                for i in fileinfo:
                    myfile1.write(i)
                    myfile1.write('\n')
        else:
            row[1] = row[1].split(';')
            l= [0 for i in range(10)]
            for  i in row[1]:
                l[int(i)] = 1
            str1 =''
            for i in l:
                str1 += ' '
                str1 += str(i)
            path = os.path.join("DataSet/train",row[0])
            for (root, dirs, files) in os.walk(path):
                imagepath =[os.path.join(root,i) for i in files]
                fileinfo = [i+ str1 for i in imagepath]
                for i in fileinfo:
                    myfile.write(i)
                    myfile.write('\n')
