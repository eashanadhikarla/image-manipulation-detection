import os
import shutil

if not os.path.exists('ProcessedData'):
    os.mkdir('ProcessedData')

trainAnnoFile = "train_filter.txt"
testAnnoFile = "test_filter.txt"

syntheticDir = "./dataset/coco_synthetic/"
realDir = "./dataset/cocostuff/coco/train2014/"
# realDir = "./dataset/train2014/"

def checkDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

checkDir('ProcessedData')
checkDir('ProcessedData/train')
checkDir('ProcessedData/test')

def moveData(realDir, syntheticDir, AnnoFile):
    if AnnoFile.startswith("train"):
        dstDir = "./ProcessedData/train"
    elif AnnoFile.startswith("test"):
        dstDir = "./ProcessedData/test"

    counter, x = 0, 0
    with open(syntheticDir+AnnoFile) as datasplit:
        filepath = datasplit.readlines()

    for f in filepath:
        eachfile = f.split(" ")
        # print(eachfile)
        # print(f"\nFilename {eachfile[0]}\nBbox x1: {eachfile[1]} y1: {eachfile[2]} x2: {eachfile[3]} y2: {eachfile[4]}\nClass label: {eachfile[5]}")
        # print(f"\nBbox x1: {eachfile[1]} y1: {eachfile[2]} x2: {eachfile[3]} y2: {eachfile[4]}\nClass label: {eachfile[5]}")

        # if (eachfile[0].startswith("Tp")) and os.path.exists(syntheticDir+eachfile[0]):
        #     # eachfilePath = syntheticDir+eachfile[0]
        #     shutil.copy(syntheticDir+eachfile[0], dstDir)
        #     counter += 1
        if eachfile[0].startswith("C")and os.path.exists(realDir+eachfile[0]):
            # eachfilePath = realDir+eachfile[0]
            shutil.copy(realDir+eachfile[0], dstDir)
            counter += 1
        else:
            print("File not found: ", eachfile[0])
            x += 1

        # if os.path.exists(eachfilePath):
            # shutil.copy(eachfilePath, dstDir)

    datasplit.close()
    return counter, x

c1, x1 = moveData(realDir, syntheticDir, trainAnnoFile)
c2, x2 = moveData(realDir, syntheticDir, testAnnoFile)
print(c1+c2)
# print(x1)

# =================================================================================

import networkx as nx
import numpy as np
import os
from glob import glob
import sys
import skimage.io as io
import pdb
import cv2
def contain_node(Graph_list,node):
        for g in Graph_list:
                if g.has_node(node):
                        return True
        return False
data_dir='../../dataset/filter_tamper' 
ext='Tp*'
dataDir='/data/MediaForensics/coco/images'
dataType='train2014' #COCO2014 train directory

cls=['person','airplane','dog','train','bed','refrigerator']
filenames=glob(os.path.join(data_dir,ext))
dDir = '../ProcessedData/train'
dDir1 = '../ProcessedData/test'

file = open('train_filter.txt','r+')
Lines = file.readlines()
counter = 1
file1 = open(os.path.join(dDir, "train.txt"), "w")

for line in Lines:
    content=line.split("_")
    content1=line.split()
    print(content)

    # if content[0] == 'Tp':
    #     target_name=content[1]
    #     source_name=content[2]
    #     source_img=io.imread(os.path.join(dataDir,dataType,'COCO_train2014_{:012d}.jpg'.format(int(source_name))))
    #     target_img=io.imread(os.path.join(dataDir,dataType,'COCO_train2014_{:012d}.jpg'.format(int(target_name))))
    #     # io.imsave(os.path.join(dDir,'train_{:012d}.jpg'.format(counter)),source_img)
    #     # counter += 1

    #     img = io.imread(content1[0])
    #     filename = 'train_'+target_name+'_'+source_name+'.jpg'
    #     io.imsave(os.path.join(dDir,'train_{:012d}.jpg'.format(counter)),img)
    #     counter += 1
        
    #     bbox = content1[1]+ " "+content1[2]+ " "+content1[3]+ " "+content1[4]+ " "+ "tamped\n"
    #     file1.write(bbox)
    #     io.imsave(os.path.join(dDir,'train_{:012d}.jpg'.format(counter)),target_img)
    #     counter += 1

    # else:
    #     bbox = content1[1]+ " "+content1[2]+ " "+content1[3]+ " "+content1[4]+ " "+ "target\n"
    #     file1.write(bbox)
file1.close()