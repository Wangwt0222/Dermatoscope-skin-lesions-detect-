import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import numpy as np
import os
import sys
import csv
import time
import logging
import errno
import random
import numbers
import pickle
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Process
import torch.utils.model_zoo as model_zoo
import math
from torchvision import models, transforms, utils
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from decimal import Decimal
from PIL import Image
from pathlib import Path
from torchsummary import summary

#填写cuda使用范围[0,1,2,3]
numGPUs = [2]
cuda_str = ""
for i in range(len(numGPUs)):
    cuda_str = cuda_str + str(numGPUs[i])
    if i is not len(numGPUs)-1:
        cuda_str = cuda_str + ","
print("Devices to use:",cuda_str)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str


#磁盘，选用第一块作为基底
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#标签及图片路径，5折交叉分割文件
# img_dir = '/media/scw4750/disk/test/HSV/'
# root = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/new_task3/labels/HAM10000/'
# img_dir = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/new_task3/images/HAM10000/'

#root = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/task3/labels/HAM10000/'
#img_dir = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/task3/images/HAM10000/'
#file = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/saved_model/indices_new.pkl'

root = '/media/scw4750/disk/test/skin_detect/isic2018-master/task3/labels/HAM10000/'
img_dir = '/media/scw4750/disk/test/skin_detect/isic2018-master/task3/images/HAM10000/'
file = '/media/scw4750/disk/test/skin_detect/isic2018-master/saved_model/indices_new.pkl'

#图片路径列表
if os.path.isdir(img_dir):
    task3_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(img_dir)
                       if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]

    task3_image_ids.sort()
# print(task3_image_ids)
im_paths = []
for image_id in task3_image_ids:
    #得到jpg结尾的文件，放入列表
    img_fname = image_id + '.jpg'
    img_fname = os.path.join(img_dir, img_fname)
    im_paths.append(img_fname)
# im_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
# print("im_paths", np.array(im_paths).shape)

#打开标签文件，保存到字典中
labels_dict = {}
with open(root+'label.csv', newline='') as csvfile:
    labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in labels_str:
        if 'ISIC' not in row[0]:
            continue
        labels_dict[row[0]] = np.array(
            [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
             int(float(row[5])), int(float(row[6])), int(float(row[7]))])
# print("labels_dict:", labels_dict)

#图片id列表以及标签列表
img_ids_list = []
labels_list = []
for img in im_paths:
    id = img[img.rindex("/") + 1:img.rindex(".")]
    array = labels_dict.get(id)
    img_ids_list.append(id)
    labels_list.append(array)
# print("img_ids_list:", img_ids_list)
# print("labels_list:", labels_list)

#标签数组以及计算各类所占比例
labels_array = np.zeros([len(labels_list), 7], dtype=np.float32)
for i in range(len(labels_list)):
    labels_array[i, :] = labels_list[i]
# print("labels_array:", labels_array)
# print(np.mean(labels_array, axis=0))

#保存5折交叉验证的标签
with open(file, 'rb') as f:
    indices = pickle.load(f)
trainIndCV = indices['trainIndCV']
valIndCV = indices['valIndCV']
#最终得到的最优值
f1Best = {}
sensBest = {}
specBest = {}
accBest = {}
waccBest = {}
aucBest = {}
convergeTime = {}
bestPred = {}
target = {}

#用到的几个常数
# cv = 0
lr = 0.001*len(numGPUs)
lastBestInd = -1
batchSize = 20 * len(numGPUs)
start_epoch = 1
display_step = 5
training_steps = 150
multiCropEval = 36
input_size = [224,224,3]
input_size_load = [450,600,3]
eval_set = 'valInd'

# Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
cropPositions = np.zeros([multiCropEval,2],dtype=np.int64)
ind = 0
for i in range(np.int32(np.sqrt(multiCropEval))):
    for j in range(np.int32(np.sqrt(multiCropEval))):
        cropPositions[ind,0] = input_size[0]/2+i*((input_size_load[0]-input_size[0])/(np.sqrt(multiCropEval)-1))
        cropPositions[ind,1] = input_size[1]/2+j*((input_size_load[1]-input_size[1])/(np.sqrt(multiCropEval)-1))
        ind += 1
print("Positions",cropPositions)
test_im = np.zeros(input_size_load)
height = input_size[0]
width = input_size[1]
for i in range(multiCropEval):
    im_crop = test_im[np.int32(cropPositions[i,0]-height/2):np.int32(cropPositions[i,0]-height/2)+height,np.int32(cropPositions[i,1]-width/2):np.int32(cropPositions[i,1]-width/2)+width,:]
    print("Shape",i+1,im_crop.shape) 

                

#要记录的内容
def log_variable(var_name, var_value):
    print('{: <20} : {}'.format(var_name, var_value))

def get_run_dir(run_name, cv):
    dirname = os.path.join(model_data_dir, run_name, 'fold' + str(cv))
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return dirname
    
def get_log_filename(run_name, cv):
    dirname = get_run_dir(run_name, cv)
    log_filename = os.path.join(dirname, '%s_log.txt' % run_name)
    return log_filename

class Tee(object):
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        return callable1(*args, **kwargs)


#数据类，特别重要
class ISICdataset(Dataset):
    def __init__(self, setInd, im_paths, labels_array, train=True):
        self.train = train
        self.same_sized_crop = True
        self.full_color_distort = True
        self.input_size = (np.int32([224, 224, 3][0]),np.int32([224, 224, 3][1]))
        self.setMean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
        self.indices = setInd
        self.im_paths = im_paths
        self.labels_array = labels_array

        if self.train:
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
                # Color distortion
            if self.full_color_distort:
                color_distort = transforms.ColorJitter(brightness=32. / 255., saturation=0.5, contrast=0.5, hue=0.2)
            else:
                color_distort = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
                # All transforms
            self.composed = transforms.Compose([
                cropping,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
#                 color_distort,
                transforms.ToTensor(),
                transforms.Normalize(torch.from_numpy(np.array([0.7811761,0.5264199,0.54028153]).astype(np.float32)).float(),
                                     torch.from_numpy(np.array([0.13406543,0.17649162,0.1900084])).float())
            ])
            self.labels = labels_array[setInd, :]
            self.im_paths = np.array(im_paths)[setInd].tolist()
        else:
            inds_rep = np.repeat(setInd, multiCropEval)
            self.labels = labels_array[inds_rep,:]
            # Path to images for loading, only for current indSet, repeat for multiordercrop
            self.im_paths = np.array(im_paths)[inds_rep].tolist()
            print(len(self.im_paths))
            # Set up crop positions for every sample
            self.cropPositions = np.tile(cropPositions, (setInd.shape[0],1))
            print("CP",self.cropPositions.shape)          
            # Set up transforms
            self.norm = transforms.Normalize(torch.from_numpy(np.array([0.7811761,0.5264199,0.54028153]).astype(np.float32)).float(),
                                             torch.from_numpy(np.array([0.13406543,0.17649162,0.1900084])).float())
            self.trans = transforms.ToTensor()            

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x = Image.open(self.im_paths[idx])
        y = self.labels[idx, :]
        if self.train:
            x = self.composed(x)
        else:
            # Apply ordered cropping to validation or test set
            # First, to pytorch tensor (0.0-1.0)
            x = self.trans(x)
            # Normalize
            x = self.norm(x)
            # Get current crop position
            x_loc = self.cropPositions[idx,0]
            y_loc = self.cropPositions[idx,1]
            # Then, apply current crop
            x = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]       
        y = np.argmax(y)
        y = np.int64(y)
        return x, y, idx
    
    
#计算各种评价指标参数
def getErrClassification_mgpu(indices, batchSize, numGPUs, device, train=False):
    # Set up sizes
    if train:
        numBatches = int(math.floor(len(indices)/batchSize/len(numGPUs)))
    else:
        numBatches = int(math.ceil(len(indices)/batchSize/len(numGPUs)))
    loss_all = np.zeros([numBatches])
    #预测值(2006,7)
    predictions = np.zeros([len(indices),7])
    #期望值(2006,7)
    targets = np.zeros([len(indices),7])
    # 验证时采用多尺度裁剪
    print('----------------------multiCropEval-----------------------------')
    loss_mc = np.zeros([len(indices)])
    predictions_mc = np.zeros([len(indices),7,36])
    targets_mc = np.zeros([len(indices),7,36])   
    for i, (inputs, labels, inds) in enumerate(valloader):
        # Get data
        inputs = inputs.to(device)
        labels = labels.to(device)       
        # Not sure if thats necessary
        optimizer.zero_grad()    
        with torch.set_grad_enabled(False):
            # Get outputs
            outputs = model(inputs)
            preds = softmax(outputs)      
            # Loss
            loss = criterion(outputs, labels)
        # Write into proper arrays
        loss_mc[i] = np.mean(loss.cpu().numpy())
#             print('preds:',preds.shape)
#             print('predictions_mc:', predictions_mc.shape)
        predictions_mc[i,:,:] = np.transpose(preds)
        tar_not_one_hot = labels.data.cpu().numpy()
        tar = np.zeros((tar_not_one_hot.shape[0], 7))
        tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
        targets_mc[i,:,:] = np.transpose(tar)
    # Targets stay the same
    targets = targets_mc[:,:,0]
    voting_scheme = 'average'
    if voting_scheme == 'vote':
        # Vote for correct prediction
        print("Pred Shape",predictions_mc.shape)
        predictions_mc = np.argmax(predictions_mc,1)    
        print("Pred Shape",predictions_mc.shape) 
        for j in range(predictions_mc.shape[0]):
            predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=7)   
        print("Pred Shape",predictions.shape) 
    elif voting_scheme == 'average':
        predictions = np.mean(predictions_mc,2)    
        
#     for i, (inputs, labels, valindices) in enumerate(valloader):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         with torch.set_grad_enabled(False):
#             outputs = model(inputs)
#             preds = softmax(outputs)
#             loss = criterion(outputs, labels)
#         bSize = batchSize
#         loss_all[i*bSize:(i+1)*bSize] = loss
#         predictions[i*bSize:(i+1)*bSize,:] = preds
#         tar_not_one_hot = labels.data.cpu().numpy()
#         tar = np.zeros((tar_not_one_hot.shape[0], 7))
#         tar[np.arange(tar_not_one_hot.shape[0]), tar_not_one_hot] = 1
#         targets[i*bSize:(i+1)*bSize,:] = tar

    # Accuarcy正确率
    acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
    # Confusion matrix混淆矩阵
    conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1))
    if conf.shape[0] < 7:
        conf = np.ones([7,7])
    # Class weighted accuracy类加权正确率
    wacc = conf.diagonal()/conf.sum(axis=1)
    # Sensitivity / Specificity敏感性/特异性
    sensitivity = np.zeros([7])
    specificity = np.zeros([7])
    #超过2类时，各种测量值计算方式
    for k in range(7):
        sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
        true_negative = np.delete(conf,[k],0)
        true_negative = np.delete(true_negative,[k],1)
        true_negative = np.sum(true_negative)
        false_positive = np.delete(conf,[k],0)
        false_positive = np.sum(false_positive[:,k])
        specificity[k] = true_negative/(true_negative+false_positive)
        f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')
    # AUC
    fpr = {}
    tpr = {}
    roc_auc = np.zeros([7])
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
#     return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc
    return loss_mc, acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc 

#5折交叉的数据，选第一折
# for cv in range(5):
if __name__ == "__main__":
    cv = 2
    print('Train')
    trainInd = trainIndCV[cv]
    print(trainInd.shape)
    print('val')
    valInd = valIndCV[cv]
    print(valInd.shape)

    #不平衡类加权方法
    indices_ham = trainInd[trainInd < 10015]
    class_weights = 1.0 / np.mean(labels_array[indices_ham, :], axis=0)
    print("Current class weights", class_weights)


    #训练集准备
    trainset = ISICdataset(trainInd, im_paths, labels_array, train=True)
    #测试集准备
    valset = ISICdataset(valInd, im_paths, labels_array, train=False)
    #数据加载
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=multiCropEval, shuffle=False, num_workers=2, pin_memory=True)

    #预训练模型
    model = models.resnet101(pretrained=True)
#     model.zero_init_residual=True
# #     #修改最后分类层
#     class model_bn(nn.Module):
#         def __init__(self, model, feature_size=2048):

#             super(model_bn, self).__init__() 
#             self.features = nn.Sequential(*list(model.children())[:-1])
# #             self.num_ftrs = model.classifier.in_features
#             self.num_ftrs = model.fc.in_features
#             self.classifier = nn.Sequential(
#                 nn.BatchNorm1d(self.num_ftrs),
#                 nn.Dropout(0.5),
#                 nn.Linear(self.num_ftrs, feature_size),
#                 nn.BatchNorm1d(feature_size),
#                 nn.ELU(inplace=True),
#                 nn.Dropout(0.5),
#                 nn.Linear(feature_size, 7),
#             )
#         def forward(self, x):
#             x = self.features(x)
#             x = x.view(x.size(0), -1)
#             x = self.classifier(x)
#             return x
    
#     model = model_bn(model)       
#     model.fc = nn.Linear(2048, 7)
    
#     summary(model.to(device),(3,224,224))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)

    if len(numGPUs) > 1:
        model = nn.DataParallel(model)
    #cuda训练
    model.cuda()
    #交叉损失函数
    criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
    #优化函数
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)   

    #学习率调整依据
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=1/np.float32(2))
    #softmax多分类
    softmax = nn.Softmax(dim=1)
    #调用cuda
    # criterion.cuda()


    #模型保存位置
    model_data_dir = "/media/scw4750/disk/wwt/2019_7_25/model_data/"
    #模型名字
    run_name = 'resnet101_no_MLP'
    log_filename = get_log_filename(run_name,cv)
    logfile = open(log_filename, 'w+')   
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)            
    # Run training
    valBest = 1000
    start_time = time.time()
    print("Start training...")
    for step in range(start_epoch, training_steps + 1):
        if step >= 50 - 25:
            scheduler.step()
        model.train()
        for j, (inputs, labels, indices) in enumerate(trainloader):
            inputs = inputs.cuda()
#             print(inputs)
#             print(inputs.shape)
            labels = labels.cuda()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
#                 print("[%d|%d] loss:%f" % (j, len(trainloader), loss.mean()))
        if step % 5 == 0 or step == 1:
            duration = time.time() - start_time
            model.eval()
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, Roc_Auc, waccuracy, predictions, targets, _ = \
                getErrClassification_mgpu(valInd, batchSize, numGPUs, device, train=False)
            eval_metric = -np.mean(waccuracy)
            if eval_metric < valBest:
                valBest = eval_metric
                f1Best[cv] = f1
                sensBest[cv] = sensitivity
                specBest[cv] = specificity
                accBest[cv] = accuracy
                waccBest[cv] = waccuracy
                aucBest[cv] = Roc_Auc
                oldBestInd = lastBestInd
                lastBestInd = step
                convergeTime[cv] = step
                # Save best predictions
                bestPred[cv] = predictions
                target[cv] = targets
                # Delte previously best model
                if os.path.isfile(get_run_dir(run_name,cv) + '/checkpoint_best-' + str(oldBestInd) + '.pt'):
                    os.remove(get_run_dir(run_name,cv) + '/checkpoint_best-' + str(oldBestInd) + '.pt')
                # Save currently best model
                state = {'epoch': step, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, get_run_dir(run_name,cv) + '/checkpoint_best-' + str(step) + '.pt')

                # If its not better, just save it delete the last checkpoint if it is not current best one
            # Save current model
            state = {'epoch': step, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, get_run_dir(run_name,cv) + '/checkpoint-' + str(step) + '.pt')
            # Delete last one
            if step == display_step:
                lastInd = 1
            else:
                lastInd = step - display_step
            if os.path.isfile(get_run_dir(run_name,cv) + '/checkpoint-' + str(lastInd) + '.pt'):
                os.remove(get_run_dir(run_name,cv) + '/checkpoint-' + str(lastInd) + '.pt')
            # Print
            print("\n")
            print('Fold: %d Epoch: %d/%d (%d h %d m %d s)' % (
            cv, step, training_steps, int(duration / 3600), int(np.mod(duration, 3600) / 60),
            int(np.mod(np.mod(duration, 3600), 60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
            print("Loss on ", eval_set, "set: ", round(Decimal(np.mean(loss)),4), " Accuracy: ", round(Decimal(accuracy),4), " F1: ", round(Decimal(f1),4))
            print(" (best WACC: ", round(Decimal(-valBest),4), " at Epoch ", lastBestInd, ")")
            print("Auc", np.around(Roc_Auc, decimals=4))
            print("Mean AUC", round(Decimal(np.mean(Roc_Auc)),4))
            print("Per Class Acc:", np.around(waccuracy, decimals=4))
            print("Weighted Accuracy:", round(Decimal(np.mean(waccuracy)),4))
            print("Sensitivity: ", np.around(sensitivity, decimals=4))
            print("Specificity:", np.around(specificity, decimals=4))
            print("Mean Spec:", np.around(np.mean(specificity), decimals=4))
            print("Confusion Matrix")
            print(conf_matrix)

    print("Best F1:", round(Decimal(f1Best[cv]),4))
    print("Best Acc:", round(Decimal(accBest[cv]),4))
    print("Best Per Class Accuracy:", np.around(waccBest[cv], decimals=4))
    print("Best Weighted Acc:", round(Decimal(np.mean(waccBest[cv])),4))
    print("Best Sens:", np.around(sensBest[cv], decimals=4))
    print("Best Spec:", np.around(specBest[cv], decimals=4))
    print("Best Mean Spec:", np.around(np.mean(specBest[cv]), decimals=4))        
    print("Best AUC:", np.around(aucBest[cv], decimals=4))
    print("Best Mean AUC:", round(Decimal(np.mean(aucBest[cv])),4))
    print("Convergence Steps:", convergeTime[cv])
    sys.stdout = original