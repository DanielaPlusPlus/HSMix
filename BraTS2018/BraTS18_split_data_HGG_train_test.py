
import os
import random
import shutil
from shutil import copy2


"""制作类别图像的训练集，和测试集所需要的文件夹，每个文件夹含二级路径"""
def mkTotalDir(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # dic=['train','val','test']
    # for i in range(0,3):
    #     current_path=data_path+dic[i]+'/'
    #     #这个函数用来判断当前路径是否存在，如果存在则创建失败，如果不存在则可以成功创建
    #     isExists=os.path.exists(current_path)
    #     if not isExists:
    #         os.makedirs(current_path)
    #         print('successful '+dic[i])
    #     else:
    #         print('is existed')
    # return

"""
source_path:原始多类图像的存放路径
train_path:训练集图像的存放路径
test_path:测试集图像的存放路径
train : test = 8 : 2
"""
def divideTrainValidationTest(source_path,train_path,test_path):
    source_image_dir=os.listdir(source_path)
    random.shuffle(source_image_dir)
    test_image_list=source_image_dir[0:int(0.2*len(source_image_dir))]
    train_image_list=source_image_dir[int(0.2*len(source_image_dir)):]

    for train_image in train_image_list:
        origins_train_image_path = source_path+'/'+train_image+'/'
        new_image_path = train_path + '/' + train_image
        isExists=os.path.exists(new_image_path)
        if not isExists:
            os.makedirs(new_image_path)
        file_dir = os.listdir(origins_train_image_path)
        for i in range(0,len(file_dir)):
            source_image = source_path + '/' + train_image + '/' + file_dir[i]
            copy2(source_image,new_image_path)
    for test_image in test_image_list:
        origins_test_image_path = source_path+'/'+test_image+'/'
        new_image_path = test_path + '/' + test_image
        isExists=os.path.exists(new_image_path)
        if not isExists:
            os.makedirs(new_image_path)
        file_dir = os.listdir(origins_test_image_path)
        for i in range(0,len(file_dir)):
            source_image = source_path + '/' + test_image + '/' + file_dir[i]
            copy2(source_image,new_image_path)


""""生成测试集、验证集、测试集的txt文件"""
def generatetxt(train_path,test_path):

    files_train = os.listdir(train_path)
    files_test = os.listdir(test_path)

    train = open('./data/BraTS2018_split/train_HGG.txt', 'a')
    test = open('./data/BraTS2018_split/test_HGG.txt', 'a')

    for file in files_train:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name =file + ' '  +  '\n'
        train.write(name)
    for file in files_test:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name =file + ' '  +  '\n'
        test.write(name)

    train.close()
    test.close()


if __name__=='__main__':
    data_path = './data/BraTS2018_split'#划分以后的train.val.test图像文件夹的存放位置
    source_path = './data/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG'#划分前所有图像文件夹的存放位置（文件夹的存储层级是一级，按照标签命名）
    train_path = './data/BraTS2018_split/train/HGG'#划分以后训练集图像对应的存放位置
    test_path = './data/BraTS2018_split/test/HGG'#划分以后测试集图像对应的存放位置

    mkTotalDir(data_path)#按照路径建立训练集/测试集/验证集划分后的文件夹
    mkTotalDir(train_path)
    mkTotalDir(test_path)


    divideTrainValidationTest(source_path,train_path, test_path)#整体列表打乱后划分，并将图像移动到对应文件夹
    generatetxt(train_path,test_path)#根据对应划分后的图片文件夹，生成对应的txt文件

