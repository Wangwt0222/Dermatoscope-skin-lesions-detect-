#对16份裁剪后的向量用SVM分类器以及随机森林分类，效果有一定程度提升
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
import numbers
import random
import pickle
import pretrainedmodels
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from multiprocessing import Process
import torch.utils.model_zoo as model_zoo
import math
from torchvision import models, transforms, utils
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from decimal import Decimal
from PIL import Image

# # 填写cuda使用范围[0,1,2,3]
# numGPUs = [0]
# cuda_str = ""
# for i in range(len(numGPUs)):
#     cuda_str = cuda_str + str(numGPUs[i])
#     if i is not len(numGPUs) - 1:
#         cuda_str = cuda_str + ","
# print("Devices to use:", cuda_str)
# os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str

# # 磁盘，选用第一块作为基底
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # 标签及图片路径，5折交叉分割文件
# # root = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/new_task3/labels/HAM10000/'
# # img_dir = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/new_task3/images/HAM10000/'
# # img_dir = '/home/deeplearning/wwt/weight_test/crop224/'
# root = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/task3/labels/HAM10000/'
# img_dir = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/task3/images/HAM10000/'
# file = '/home/deeplearning/wwt/pymodel/model/skin_detect/isic2018-master/saved_model/indices_new.pkl'

# #图片路径列表
# if os.path.isdir(img_dir):
#     task3_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(img_dir)
#                        if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]

#     task3_image_ids.sort()
# # print(task3_image_ids)
# im_paths = []
# for image_id in task3_image_ids:
#     #得到jpg结尾的文件，放入列表
#     img_fname = image_id + '.jpg'
#     img_fname = os.path.join(img_dir, img_fname)
#     im_paths.append(img_fname)
# # im_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
# # print("im_paths", np.array(im_paths).shape)

# # 打开标签文件，保存到字典中
# labels_dict = {}
# with open(root + 'label.csv', newline='') as csvfile:
#     labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in labels_str:
#         if 'ISIC' not in row[0]:
#             continue
#         labels_dict[row[0]] = np.array(
#             [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
#              int(float(row[5])), int(float(row[6])), int(float(row[7]))])
# # print("labels_dict:", labels_dict)

# # 图片id列表以及标签列表
# img_ids_list = []
# labels_list = []
# for img in im_paths:
#     id = img[img.rindex("/") + 1:img.rindex(".")]
#     array = labels_dict.get(id)
#     img_ids_list.append(id)
#     labels_list.append(array)
# # print("img_ids_list:", img_ids_list)
# # print("labels_list:", labels_list)

# # 标签数组以及计算各类所占比例
# labels_array = np.zeros([len(labels_list), 7], dtype=np.float32)
# for i in range(len(labels_list)):
#     labels_array[i, :] = labels_list[i]
# # print("labels_array:", labels_array)
# # print(np.mean(labels_array, axis=0))

# # 保存5折交叉验证的标签
# with open(file, 'rb') as f:
#     indices = pickle.load(f)
# trainIndCV = indices['trainIndCV']
# # valIndCV = indices['valIndCV']

# # 用到的几个常数
# # cv = 0
# lr = 0.000025 * len(numGPUs)
# lastBestInd = -1
# batchSize = 20 * len(numGPUs)
# start_epoch = 1
# display_step = 5
# training_steps = 15
# multiCropEval = 16
# input_size = [224,224,3]
# input_size_load = [450,600,3]
# eval_set = 'valInd'

# # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
# cropPositions = np.zeros([multiCropEval,2],dtype=np.int64)
# ind = 0
# for i in range(np.int32(np.sqrt(multiCropEval))):
#     for j in range(np.int32(np.sqrt(multiCropEval))):
#         cropPositions[ind,0] = input_size[0]/2+i*((input_size_load[0]-input_size[0])/(np.sqrt(multiCropEval)-1))
#         cropPositions[ind,1] = input_size[1]/2+j*((input_size_load[1]-input_size[1])/(np.sqrt(multiCropEval)-1))
#         ind += 1
# # Sanity checks
# print("Positions",cropPositions)
# # Test image sizes
# test_im = np.zeros(input_size_load)
# height = input_size[0]
# width = input_size[1]
# for i in range(multiCropEval):
#     im_crop = test_im[np.int32(cropPositions[i,0]-height/2):np.int32(cropPositions[i,0]-height/2)+height,np.int32(cropPositions[i,1]-width/2):np.int32(cropPositions[i,1]-width/2)+width,:]
#     print("Shape",i+1,im_crop.shape)

# class RandomCrop224(object):

#     def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.padding = padding
#         self.pad_if_needed = pad_if_needed
#         self.fill = fill
#         self.padding_mode = padding_mode

#     @staticmethod
#     def get_params(img, output_size):

#         w, h = img.size
#         th, tw = output_size
#         if w == tw and h == th:
#             return 0, 0, h, w

#         random.seed(0)
#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)
#         return i, j, th, tw

#     def __call__(self, img):

#         i, j, h, w = self.get_params(img, self.size)

#         return img.crop((j, i, j + w, i + h))

#     def __repr__(self):
#         return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)



# # 数据类，特别重要
# class ISICdataset(Dataset):
#     def __init__(self, setInd, im_paths, labels_array, train=True):
#         self.train = train
#         self.same_sized_crop = True
#         self.full_color_distort = False
#         self.input_size = (np.int32([224, 224, 3][0]),np.int32([224, 224, 3][1]))
#         self.setMean = np.array([0, 0, 0]).astype(np.float32)
#         self.indices = setInd
#         self.im_paths = im_paths
#         self.labels_array = labels_array

#         if self.train:
#             if self.same_sized_crop:
#                 cropping = transforms.RandomCrop(self.input_size)
#             else:
#                 cropping = transforms.RandomResizedCrop(self.input_size[0])
#                 # Color distortion
#             if self.full_color_distort:
#                 color_distort = transforms.ColorJitter(brightness=32. / 255., saturation=0.5, contrast=0.5, hue=0.2)
#             else:
#                 color_distort = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
#                 # All transforms
#             self.composed = transforms.Compose([
#                 cropping,
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 color_distort,
#                 transforms.ToTensor(),
#                 transforms.Normalize(torch.from_numpy(self.setMean).float(),
#                                      torch.from_numpy(np.array([1., 1., 1.])).float())
#             ])
#             self.labels = labels_array[setInd, :]
#             self.im_paths = np.array(im_paths)[setInd].tolist()
#         else:
#             inds_rep = np.repeat(setInd, multiCropEval)
#             self.labels = labels_array[inds_rep,:]
#             # Path to images for loading, only for current indSet, repeat for multiordercrop
#             self.im_paths = np.array(im_paths)[inds_rep].tolist()
#             print(len(self.im_paths))
#             # Set up crop positions for every sample
#             self.cropPositions = np.tile(cropPositions, (np.array(setInd).shape[0],1))
#             print("CP",self.cropPositions.shape)          
#             # Set up transforms
#             self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
#             self.trans = transforms.ToTensor()            

#     def __len__(self):
#         return self.labels.shape[0]

#     def __getitem__(self, idx):
#         x = Image.open(self.im_paths[idx])
#         y = self.labels[idx, :]
#         if self.train:
#             x = self.composed(x)
#         else:
#             # Apply ordered cropping to validation or test set
#             # First, to pytorch tensor (0.0-1.0)
#             x = self.trans(x)
#             # Normalize
#             x = self.norm(x)
#             # Get current crop position
#             x_loc = self.cropPositions[idx,0]
#             y_loc = self.cropPositions[idx,1]
#             # Then, apply current crop
#             x = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]       
#         y = np.argmax(y)
#         y = np.int64(y)
#         return x, y, idx


# # 计算各种评价指标参数
# def getErrClassification_mgpu(indices, batchSize, numGPUs, device, train=False):
#     # Set up sizes
#     if train:
#         numBatches = int(math.floor(len(indices) / batchSize / len(numGPUs)))
#     else:
#         numBatches = int(math.ceil(len(indices) / batchSize / len(numGPUs)))
#     loss_all = np.zeros([numBatches])
#     # 预测值(2006,7)
#     predictions = np.zeros([len(indices), 7])
#     # 期望值(2006,7)
#     targets = np.zeros([len(indices), 7])
#     print('----------------------multiCropEval-----------------------------')
#     loss_mc = np.zeros([len(indices)])
#     predictions_mc = np.zeros([len(indices),7,16])
#     targets_mc = np.zeros([len(indices),7,16])   
#     for i, (inputs, labels, inds) in enumerate(valloader):
#         # Get data
#         inputs = inputs.to(device)
#         labels = labels.to(device)       
#         # Not sure if thats necessary
#         optimizer.zero_grad()    
#         with torch.set_grad_enabled(False):
#             # Get outputs
#             outputs = model(inputs)
#             preds = softmax(outputs)      
#             # Loss
#             loss = criterion(outputs, labels) 
#         # Write into proper arrays
#         loss_mc[i] = np.mean(loss.cpu().numpy())
# #             print('preds:',preds.shape)
# #             print('predictions_mc:', predictions_mc.shape)
#         predictions_mc[i,:,:] = np.transpose(preds)
#         tar_not_one_hot = labels.data.cpu().numpy()
#         tar = np.zeros((tar_not_one_hot.shape[0], 7))
#         tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
#         targets_mc[i,:,:] = np.transpose(tar)
#     # Targets stay the same
#     targets = targets_mc[:,:,0]
#     voting_scheme = 'average'
#     if voting_scheme == 'vote':
#         # Vote for correct prediction
#         print("Pred Shape",predictions_mc.shape)
#         predictions_mc = np.argmax(predictions_mc,1)    
#         print("Pred Shape",predictions_mc.shape) 
#         for j in range(predictions_mc.shape[0]):
#             predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
#         print("Pred Shape",predictions.shape) 
#     elif voting_scheme == 'average':
#         predictions = np.mean(predictions_mc,2)
# #     # 验证时采用多尺度裁剪
# #     for i, (inputs, labels, valindices) in enumerate(valloader):
# #         inputs = inputs.to(device)
# #         labels = labels.to(device)
# #         optimizer.zero_grad()
# #         with torch.set_grad_enabled(False):
# #             outputs = model(inputs)
# #             preds = softmax(outputs)
# #             loss = criterion(outputs, labels)
# #         bSize = batchSize
# #         loss_all[i * bSize:(i + 1) * bSize] = loss
# #         predictions[i * bSize:(i + 1) * bSize, :] = preds
# #         tar_not_one_hot = labels.data.cpu().numpy()
# #         tar = np.zeros((tar_not_one_hot.shape[0], 7))
# #         tar[np.arange(tar_not_one_hot.shape[0]), tar_not_one_hot] = 1
# #         targets[i * bSize:(i + 1) * bSize, :] = tar

#     # Accuarcy正确率
#     acc = np.mean(np.equal(np.argmax(predictions, 1), np.argmax(targets, 1)))
#     # Confusion matrix混淆矩阵
#     conf = confusion_matrix(np.argmax(targets, 1), np.argmax(predictions, 1))
#     if conf.shape[0] < 7:
#         conf = np.ones([7, 7])
#     # Class weighted accuracy类加权正确率
#     wacc = conf.diagonal() / conf.sum(axis=1)
#     # Sensitivity / Specificity敏感性/特异性
#     sensitivity = np.zeros([7])
#     specificity = np.zeros([7])
#     # 超过2类时，各种测量值计算方式
#     for k in range(7):
#         sensitivity[k] = conf[k, k] / (np.sum(conf[k, :]))
#         true_negative = np.delete(conf, [k], 0)
#         true_negative = np.delete(true_negative, [k], 1)
#         true_negative = np.sum(true_negative)
#         false_positive = np.delete(conf, [k], 0)
#         false_positive = np.sum(false_positive[:, k])
#         specificity[k] = true_negative / (true_negative + false_positive)
#         f1 = f1_score(np.argmax(predictions, 1), np.argmax(targets, 1), average='weighted')
#     # AUC
#     fpr = {}
#     tpr = {}
#     roc_auc = np.zeros([7])
#     for i in range(7):
#         fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc 

if __name__ == "__main__":
#     cv=0
#     # 5折交叉的数据，选第一折
#     # for cv in range(5):

#     # print('Train')
#     trainInd = trainIndCV[cv]
#     with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/valInd2.pkl', 'rb') as f:
#         valInd0 = pickle.load(f)
# #0为val，1为test    
#     valInd = valInd0[1]

#     # 不平衡类加权方法
#     indices_ham = trainInd[trainInd < 10015]
#     class_weights = 1.0 / np.mean(labels_array[indices_ham, :], axis=0)
#     # print("Current class weights", class_weights)

#     valset = ISICdataset(valInd, im_paths, labels_array, train=False)
#     valloader = torch.utils.data.DataLoader(valset, batch_size=multiCropEval, shuffle=False, num_workers=2, pin_memory=True)

#     model = pretrainedmodels.__dict__['dpn68b'](num_classes=1000, pretrained='imagenet+5k')
# #     model = pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
# #     model = models.resnet101(pretrained=True)
#     # 修改最后分类层
# #     model.fc = nn.Linear(2048, 7)
# #     num_ftrs = model.classifier.in_features
# #     model.classifier = nn.Linear(num_ftrs, 7)
# #     num_ftrs = model.last_linear.in_features
# #     model.last_linear = nn.Linear(num_ftrs, 7)
#     num_ftrs = model.classifier.in_channels
#     model.classifier = nn.Conv2d(num_ftrs,7,[1,1])
#     model.eval()
#     # cuda训练
#     model.cuda()
#     # model.eval()
#     # 交叉损失函数
#     criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
#     # 优化函数
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     # 学习率调整依据
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=1 / np.float32(2))
#     # softmax多分类
#     softmax = nn.Softmax(dim=1)

#     start_time = time.time()
    
#     state = torch.load('/media/scw4750/disk/test/torch_crop10_28/dpn68b/fold2/checkpoint_best-55.pt')
# #     state = torch.load('/home/deeplearning/wwt/weight_test/pytorch_test/model_data/resnet101/fold0/checkpoint_best-35.pt')
#     model.load_state_dict(state['state_dict'])
#     optimizer.load_state_dict(state['optimizer'])
    
#     loss, accuracy, sensitivity, specificity, conf_matrix, f1, Roc_Auc, waccuracy, predictions, targets, predictions_mc = \
#         getErrClassification_mgpu(valInd, batchSize, numGPUs, device, train=False)
#     duration = time.time() - start_time
#     print("\n")
#     print('Fold: %d (%d h %d m %d s)' % (
#         cv, int(duration / 3600), int(np.mod(duration, 3600) / 60),
#         int(np.mod(np.mod(duration, 3600), 60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
#     print("Loss on ", eval_set, "set: ", round(Decimal(loss), 4), " Accuracy: ", round(Decimal(accuracy), 4), " F1: ",
#           round(Decimal(f1), 4))
#     print("Auc", np.around(Roc_Auc, decimals=4))
#     print("Mean AUC", round(Decimal(np.mean(Roc_Auc)), 4))
#     print("Per Class Acc:", np.around(waccuracy, decimals=4))
#     print("Weighted Accuracy:", round(Decimal(np.mean(waccuracy)), 4))
#     print("Sensitivity: ", np.around(sensitivity, decimals=4))
#     print("Specificity:", np.around(specificity, decimals=4))
#     print("Mean Spec:", np.around(np.mean(specificity), decimals=4))
#     print("Confusion Matrix")
#     print(conf_matrix)
    
#     import pickle
#     output = open('/home/deeplearning/wwt/weight_test/pytorch_test/features/dpn68b/pred_test.pkl', 'wb')
#     pickle.dump(predictions_mc, output)
#     output.close()
#     output1 = open('/home/deeplearning/wwt/weight_test/pytorch_test/features/dpn68b/tar_test.pkl', 'wb')
#     pickle.dump(targets, output1)
#     output1.close()

    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/resnet101/pred_val.pkl', 'rb') as f:
        predictions_mc1 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/resnet101/tar_val.pkl', 'rb') as f:
        targets1 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/densenet169/pred_val.pkl', 'rb') as f:
        predictions_mc2 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/densenet169/tar_val.pkl', 'rb') as f:
        targets2 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/senet154/pred_val.pkl', 'rb') as f:
        predictions_mc3 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/senet154/tar_val.pkl', 'rb') as f:
        targets3 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/se_resnext50/pred_val.pkl', 'rb') as f:
        predictions_mc4 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/se_resnext50/tar_val.pkl', 'rb') as f:
        targets4 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/dpn68b/pred_val.pkl', 'rb') as f:
        predictions_mc5 = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/dpn68b/tar_val.pkl', 'rb') as f:
        targets5 = pickle.load(f)        
    predictions_mc = np.concatenate((predictions_mc1,predictions_mc2,predictions_mc3,predictions_mc4,predictions_mc5),axis=0)
#     predictions_mc = predictions_mc1
    print(predictions_mc.shape)
    targets = np.concatenate((targets1,targets2,targets3,targets4,targets5),axis=0)
#     targets = targets1
    print(targets.shape)
                                                             
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/resnet101/pred_test.pkl', 'rb') as f:
        pred_test = pickle.load(f)
    with open('/home/deeplearning/wwt/weight_test/pytorch_test/features/resnet101/tar_test.pkl', 'rb') as f:
        tar_test = pickle.load(f)
    
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from bayes_opt import BayesianOptimization
    from sklearn.model_selection import cross_val_score
    from sklearn.externals import joblib

    pred_val = predictions_mc 
    feat_train = np.reshape(pred_val,[len(pred_val),16*7])
    tar_train = targets
    feat_val = np.reshape(pred_test,[len(pred_test),16*7])
    tar_train_not_one_hot = np.argmax(tar_train,1)
    # Train SVM/RF
#     clf = RandomForestClassifier(class_weight='balanced')
#     clf = SVC(kernel='rbf',class_weight='balanced', C=10)
#     clf = RandomForestClassifier(class_weight='balanced',n_estimators=132,max_features=0.1163,max_depth=79,min_samples_split=74)
    clf = RandomForestClassifier(class_weight='balanced',n_estimators=255,max_features=0.1385,max_depth=75,min_samples_split=100)
#     clf = SVC(class_weight='balanced')
#     clf = XGBClassifier(class_weight='balanced',n_estimators=10000,learning_rate=0.3)
#     clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
#     clf = AdaBoostClassifier(n_estimators=100)
#     clf = GradientBoostingClassifier(n_estimators=1000)


    # 贝叶斯优化
#     def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
#         val = cross_val_score(
#             RandomForestClassifier(n_estimators=int(n_estimators),
#                 min_samples_split=int(min_samples_split),
#                 max_features=min(max_features, 0.999), 
#                 max_depth=int(max_depth),
#                 random_state=2,
#                 class_weight='balanced'
#             ),
#             feat_train, tar_train_not_one_hot, 'recall_macro', cv=5
#         ).mean()
#         return val
    
#     rf_bo = BayesianOptimization(rf_cv,
#         {'n_estimators': (10, 500),
#         'min_samples_split': (2, 100),
#         'max_features': (0.1, 0.999),
#         'max_depth': (5, 80)})
#     rf_bo.maximize()
#     print(rf_bo.res['max'])
    
    
    
#     tunned_parameters = [{'kernel':['rbf'],'C':[0.1, 1, 10, 100, 1000]},{'kernel':['poly'],'C':[0.1, 1, 10, 100, 1000]},
#                          {'kernel':['sigmoid'],'C':[0.1, 1, 10, 100, 1000]},{'kernel':['linear'], 'C':[0.1, 1, 10, 100, 1000]}]
#     scores = ['recall']
#     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#     for score in scores:
#         print("# Tuning hyper-parameters for %s" % score)
#         print()
#         grid_search = GridSearchCV(clf, tunned_parameters, cv=kfold, scoring='%s_macro' % score)
#         grid_search.fit(feat_train, tar_train_not_one_hot)
#         print("Best parameters set found on development set:")
#         print()
#         print(grid_search.best_params_)
#         print()
#         print("Grid scores on development set:")
#         print()
#         means = grid_search.cv_results_['mean_test_score']
#         stds = grid_search.cv_results_['std_test_score']
#         #这里输出了各种参数在使用交叉验证的时候得分的均值和方差
#         for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#             print("%0.3f (+/-%0.03f) for %r"
#                   % (mean, std * 2, params))
#         print()

#         print("Detailed classification report:")
#         print()
#         print("The model is trained on the full development set.")
#         print("The scores are computed on the full evaluation set.")
#         print()
#         #这里是使用训练出来的最好的参数进行预测
#         predictions_not_one_hot = grid_search.predict(feat_val)
#         predictions1 = np.zeros([len(predictions_not_one_hot),7])
#         predictions1[np.arange(len(predictions_not_one_hot)), predictions_not_one_hot] = 1
        
#         print(classification_report(np.argmax(tar_test, 1), np.argmax(predictions1, 1)))
#         print()
           
    # Same for all sklearn classifiers
    clf.fit(feat_train, tar_train_not_one_hot)
    #保存模型
#     joblib.dump(clf, 'rfc.pkl')
    #加载模型
    clf1 = joblib.load('rfc.pkl')
    clf = clf1
    
    predictions_not_one_hot = clf.predict(feat_val)
    predictions1 = np.zeros([len(predictions_not_one_hot),7])
    predictions1[np.arange(len(predictions_not_one_hot)), predictions_not_one_hot] = 1        
    print("Train score",clf.score(feat_train, tar_train_not_one_hot))

    # Accuarcy正确率
    acc = np.mean(np.equal(np.argmax(predictions1, 1), np.argmax(tar_test, 1)))
    # Confusion matrix混淆矩阵
    conf = confusion_matrix(np.argmax(tar_test, 1), np.argmax(predictions1, 1))
    if conf.shape[0] < 7:
        conf = np.ones([7, 7])
    # Class weighted accuracy类加权正确率
    wacc = conf.diagonal() / conf.sum(axis=1)
    # Sensitivity / Specificity敏感性/特异性
    sensitivity = np.zeros([7])
    specificity = np.zeros([7])
    # 超过2类时，各种测量值计算方式
    for k in range(7):
        sensitivity[k] = conf[k, k] / (np.sum(conf[k, :]))
        true_negative = np.delete(conf, [k], 0)
        true_negative = np.delete(true_negative, [k], 1)
        true_negative = np.sum(true_negative)
        false_positive = np.delete(conf, [k], 0)
        false_positive = np.sum(false_positive[:, k])
        specificity[k] = true_negative / (true_negative + false_positive)
        f1 = f1_score(np.argmax(predictions1, 1), np.argmax(tar_test, 1), average='weighted')
    # AUC
    fpr = {}
    tpr = {}
    roc_auc = np.zeros([7])
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(tar_test[:, i], predictions1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
            
    print("\n")
    print(" Accuracy: ", round(Decimal(acc), 4), " F1: ",round(Decimal(f1), 4))
    print("Auc", np.around(roc_auc, decimals=4))
    print("Mean AUC", round(Decimal(np.mean(roc_auc)), 4))
    print("Per Class Acc:", np.around(wacc, decimals=4))
    print("Weighted Accuracy:", round(Decimal(np.mean(wacc)), 4))
    print("Sensitivity: ", np.around(sensitivity, decimals=4))
    print("Specificity:", np.around(specificity, decimals=4))
    print("Mean Spec:", np.around(np.mean(specificity), decimals=4))
    print("Confusion Matrix")
    print(conf)

#画混淆矩阵图    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    y_pred = np.argmax(predictions1, 1)
    y_true = np.argmax(tar_test, 1)
    print("y_pred:",y_pred)
    print("y_true:",y_true)
    labels = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 4),dpi=140)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='black', fontsize=7, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (x_val != y_val):
                #这里是绘制数字，可以对数字大小和颜色进行修改                
                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=7, va='center', ha='center')
#                 plt.text(x_val, y_val, "%0.2f" % (0.00,), color='black', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/deeplearning/wwt/weight_test/pytorch_test/examples.jpg')
    
    
#     from sklearn.metrics import confusion_matrix
#     from sklearn.metrics import recall_score
    
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

#     guess = np.argmax(predictions1, 1)
#     fact = np.argmax(tar_test, 1)
# #     classes = list(set(fact))
# #     classes.sort()
#     classes = ["0.MEL", "1.NV", "2.BCC", "3.AKIEC", "4.BKL", "5.DF", "6.VASC"]
#     confusion = confusion_matrix(guess, fact)
#     print(confusion.shape)
#     print(np.sum(confusion,axis=0))
#     new_confusion = np.zeros(shape=(confusion.shape[0], confusion.shape[1]), dtype=np.float)
#     new_confusion = confusion / (confusion.astype(np.float).sum(axis=0)[:, np.newaxis] + 0.001)
#     print(np.around(new_confusion, decimals=2))
#     new_confusion = np.around(new_confusion, decimals=2)
#     plt.imshow(new_confusion, cmap=plt.cm.Blues)
#     indices = range(len(new_confusion))
#     plt.xticks(indices, classes)
#     plt.yticks(indices, classes)
#     plt.colorbar()
#     plt.xlabel('Predicted label')
#     plt.ylabel('True label')
#     for first_index in range(len(new_confusion)):
#         for second_index in range(len(new_confusion[first_index])):
#             plt.text(first_index, second_index, new_confusion[first_index][second_index])

    
#     plt.savefig('/home/deeplearning/wwt/weight_test/pytorch_test/examples.jpg')   
