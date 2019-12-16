# -*- coding: utf-8 -*-
#打印，输出
import numpy as np
import sys
import logging
import sys
import os
import inspect
import pandas as pd

def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))
model_data_dir = os.path.join(root_dir, 'model_data')
submission_dir = os.path.join(root_dir, 'submissions')

dir_to_make = [model_data_dir, submission_dir]
mkdir_if_not_exist(dir_list=dir_to_make)
mkdir_if_not_exist(dir_list=dir_to_make)

ISIC2018_dir = os.path.join(root_dir, 'datasets', 'ISIC2018')
data_dir = os.path.join(ISIC2018_dir, 'data')
cached_data_dir = os.path.join(ISIC2018_dir, 'cache')

mkdir_if_not_exist(dir_list=[cached_data_dir])

task12_img = 'ISIC2018_Task1-2_Training_Input'
task12_validation_img = 'ISIC2018_Task1-2_Validation_Input'
task12_test_img = 'ISIC2018_Task1-2_Test_Input'

task3_img = 'ISIC2018_Task3_Training_Input'
task3_validation_img = 'ISIC2018_Task3_Validation_Input'
task3_test_img = 'ISIC2018_Task3_Test_Input'

task1_gt = 'ISIC2018_Task1_Training_GroundTruth'
task2_gt = 'ISIC2018_Task2_Training_GroundTruth_v3'
task3_gt = 'ISIC2018_Task3_Training_GroundTruth'

MEL = 0  # Melanoma
NV = 1  # Melanocytic nevus
BCC = 2  # Basal cell carcinoma
AKIEC = 3  # Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
BKL = 4  # Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
DF = 5  # Dermatofibroma
VASC = 6  # Vascular lesion

classes = [MEL, NV, BCC, AKIEC, BKL, DF, VASC]
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

task12_img_dir = os.path.join(data_dir, task12_img)
task12_validation_img_dir = os.path.join(data_dir, task12_validation_img)
task12_test_img_dir = os.path.join(data_dir, task12_test_img)

task3_img_dir = os.path.join(data_dir, task3_img)
task3_validation_img_dir = os.path.join(data_dir, task3_validation_img)
task3_test_img_dir = os.path.join(data_dir, task3_test_img)

task1_gt_dir = os.path.join(data_dir, task1_gt)
task2_gt_dir = os.path.join(data_dir, task2_gt)
task3_gt_dir = os.path.join(data_dir, task3_gt)

task12_image_ids = list()
if os.path.isdir(task12_img_dir):
    task12_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task12_img_dir)
                        if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task12_image_ids.sort()

task12_validation_image_ids = list()
if os.path.isdir(task12_validation_img_dir):
    task12_validation_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task12_validation_img_dir)
                                   if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task12_validation_image_ids.sort()

task12_test_image_ids = list()
if os.path.isdir(task12_test_img_dir):
    task12_test_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task12_test_img_dir)
                                   if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task12_test_image_ids.sort()

task3_image_ids = list()
if os.path.isdir(task3_img_dir):
    task3_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task3_img_dir)
                       if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]

    task3_image_ids.sort()

task3_validation_image_ids = list()
if os.path.isdir(task3_validation_img_dir):
    task3_validation_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task3_validation_img_dir)
                                  if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task3_image_ids.sort()

task3_test_image_ids = list()
if os.path.isdir(task3_test_img_dir):
    task3_test_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task3_test_img_dir)
                                  if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task3_test_image_ids.sort()

task3_gt_fname = 'ISIC2018_Task3_Training_GroundTruth.csv'
task3_sup_fname = 'ISIC2018_Task3_Training_LesionGroupings.csv'

task12_images_npy_prefix = 'task12_images'
task12_validation_images_npy_prefix = 'task12_validation_images'
task12_test_images_npy_prefix = 'task12_test_images'

task3_images_npy_prefix = 'task3_images'
task3_validation_images_npy_prefix = 'task3_validation_images'
task3_test_images_npy_prefix = 'task3_test_images'

task1_gt_npy_prefix = 'task1_masks'
task2_gt_npy_prefix = 'task2_masks'
task3_gt_npy_prefix = 'task3_labels'

task2_labels = ['globules',
                'milia_like_cyst',
                'negative_network',
                'pigment_network',
                'streaks.png']

ATTRIBUTE_GLOBULES = 1
ATTRIBUTE_MILIA_LIKE_CYST = 2
ATTRIBUTE_NEGATIVE_NETWORK = 3
ATTRIBUTE_PIGMENT_NETWORK = 4
ATTRIBUTE_STREAKS = 5



class PrintColors:

    GREEN = "\033[0;32m"
    BLUE = "\033[1;34m"
    RED = "\033[1;31m"

    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_COLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def on_aws():
    if 'ubuntu' in root_dir:
        return True
    return False

def print_confusion_matrix(cm, labels):
    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [12])
    # Print header
    print()
    first_cell = "True\Pred"
    print("|%{0}s|".format(columnwidth - 2) % first_cell, end="")
    for label in labels:
        print("%{0}s|".format(columnwidth -1) % label, end="")
    print()

    first_cell = "-------"
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for _ in labels:
        print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print()

    # Print rows
    for i, label1 in enumerate(labels):
        print("|%{0}s|".format(columnwidth - 2) % label1, end="")
        for j in range(len(labels)):
            cell = "%{0}.2f|".format(columnwidth-1) % cm[i, j]
            if i == len(labels) - 1 or j == len(labels) - 1:
                cell = "%{0}d|".format(columnwidth-1) % cm[i, j]
                if i == j:
                    print("%{0}s|".format(columnwidth-1) % ' ', end="")
                else:
                    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")
            elif i == j:
                print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")
            else:
                print(PrintColors.RED + cell + PrintColors.END_COLOR, end="")

        print()


def print_precision_recall(precision, recall, labels):
    columnwidth = max([len(x) for x in labels] + [12])
    # Print header
    print()
    first_cell = " "
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for label in labels:
        print("%{0}s|".format(columnwidth-1) % label, end="")
    print("%{0}s|".format(columnwidth-1) % 'MEAN', end="")
    print()

    first_cell = "-------"
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for _ in labels:
        print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print()

    # print precision
    print("|%{0}s|".format(columnwidth-2) % 'precision', end="")
    for j in range(len(labels)):
        cell = "%{0}.3f|".format(columnwidth-1) % precision[j]
        print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")

    cell = "%{0}.3f|".format(columnwidth-1) % np.mean(precision)
    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")

    print()

    # print recall
    print("|%{0}s|".format(columnwidth-2) % 'recall', end="")
    for j in range(len(labels)):
        cell = "%{0}.3f|".format(columnwidth-1) % recall[j]
        print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")

    cell = "%{0}.3f|".format(columnwidth-1) % np.mean(recall)
    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")

    print('')

#log记录
import os
import errno
# from paths import model_data_dir


def get_run_dir(run_name):
    dirname = os.path.join(model_data_dir, run_name)
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    return dirname


def get_weights_filename(run_name):
    dirname = get_run_dir(run_name)
    weights_filename = os.path.join(dirname, '%s.hdf5' % run_name)
    return weights_filename


def get_csv_filename(run_name):
    dirname = get_run_dir(run_name)
    csv_filename = os.path.join(dirname, '%s.csv' % run_name)
    return csv_filename


def get_model_image_filename(run_name):
    dirname = get_run_dir(run_name)
    filename = os.path.join(dirname, '%s.png' % run_name)
    return filename


def get_json_filename(run_name):
    dirname = get_run_dir(run_name)
    json_filename = os.path.join(dirname, '%s.json' % run_name)
    return json_filename


def get_model_config_filename(run_name):
    dirname = get_run_dir(run_name)
    config_filename = os.path.join(dirname, '%s.pkl' % run_name)
    return config_filename

# 加载训练数据
def load_training_data(task_idx,
                       output_size=None,
                       num_partitions=5,
                       idx_partition=0,
                       test_split=0.):
# 加载图片
        suffix = '' if output_size is None else '_%d' % output_size
        images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task3_images_npy_prefix, suffix))
        images = np.load(images_npy_filename)
# 加载标签
# image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
        labels = []
        with open(os.path.join(task3_gt_dir, task3_gt_fname), 'r') as f:
            for i, line in enumerate(f.readlines()[1:]):
                fields = line.strip().split(',')
                labels.append([eval(field) for field in fields[1:]])
            labels = np.stack(labels, axis=0)
        return partition_task3_data(x=images, y=labels, k=num_partitions, i=idx_partition, test_split=test_split)

# 将加载的数据按照某种划分分成训练集，测试集和验证集
def partition_task3_data(x, y, k=5, i=0, test_split=1. / 6, seed=42):
    assert isinstance(k, int) and isinstance(i, int) and 0 <= i < k

    fname = os.path.join(task3_gt_dir, task3_sup_fname)
    assert os.path.exists(fname)

    df = pd.read_csv(os.path.join(task3_gt_dir, task3_sup_fname))
    grouped = df.groupby('lesion_id', sort=True)
    lesion_ids = []
    for name, group in grouped:
        image_ids = group.image.tolist()
        lesion_ids.append([name, image_ids])
    # shuffle lesion ids
    np.random.seed(seed)
    n = len(lesion_ids)
    indices = np.random.permutation(n)

    image_ids = [image_id for idx in indices for image_id in lesion_ids[idx][1]]
    n = len(image_ids)
    n_set = int(n * (1. - test_split)) // k
    # divide the data into (k + 1) sets, -1 is test set, [0, k) are for train and validation
    indices = [i for i in range(k) for _ in range(n_set)] + [-1] * (n - n_set * k)

    indices = list(zip(indices, image_ids))
    indices.sort(key=lambda x: x[1])
    indices = np.array([idx for idx, image_id in indices], dtype=np.uint)

    valid_indices = (indices == i)
    test_indices = (indices == -1)
    train_indices = ~(valid_indices | test_indices)

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import compute_class_weight as sk_compute_class_weight
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.utils.vis_utils import plot_model

def get_confusion_matrix(y_true, y_pred, norm_cm=True, print_cm=True):
    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)

    cnf_mat = confusion_matrix(true_class, pred_class, labels=classes)

    total_cnf_mat = np.zeros(shape=(cnf_mat.shape[0] + 1, cnf_mat.shape[1] + 1), dtype=np.float)
    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    for i_row in range(cnf_mat.shape[0]):
        total_cnf_mat[i_row, -1] = np.sum(total_cnf_mat[i_row, 0:-1])

    for i_col in range(cnf_mat.shape[1]):
        total_cnf_mat[-1, i_col] = np.sum(total_cnf_mat[0:-1, i_col])

    if norm_cm:
        cnf_mat = cnf_mat/(cnf_mat.astype(np.float).sum(axis=1)[:, np.newaxis] + 0.001)

    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    if print_cm:
        print_confusion_matrix(cm=total_cnf_mat, labels=class_names + ['TOTAL', ])

    return cnf_mat

def get_precision_recall(y_true, y_pred, print_pr=True):

    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(y_true=true_class,
                                                              y_pred=pred_class,
                                                              labels=classes,
                                                              warn_for=())
    if print_pr:
        print_precision_recall(precision=precision, recall=recall, labels=class_names)

    return precision, recall


# 回调函数，显示验证预测效果，构建混淆矩阵
class ValidationPrediction(Callback):
    def __init__(self, show_confusion_matrix=False, **kwargs):
        super(ValidationPrediction, self).__init__()

        self.show_confusion_matrix = show_confusion_matrix

        self.visualize = kwargs.get('visualize', False)
        self.nrows = kwargs.get('nrows', 5)
        self.ncols = kwargs.get('ncols', 5)
        self.mask_colors = kwargs.get('mask_colors', ['r', 'b', 'g', 'c', 'm', 'y'])
        self.n_choices = self.nrows * self.ncols

        # for display purposes
        self.fig = None
        self.ax = None
        self.indices = None

        self.confusion_fig = None
        self.confusion_ax = None

        # setup
        self.y_true = None
        self.y_pred = None

    def on_epoch_end(self, epoch, logs=None):
        self.make_predictions()
        if self.show_confusion_matrix:
            self.view_confusion_matrix()

        if self.visualize:
            self.visualize_validation_prediction()

    def make_predictions(self):
        self.y_pred = self.model.predict(self.validation_data[0])
        self.y_true = self.validation_data[1]

    def view_confusion_matrix(self):
        _ = get_confusion_matrix(y_true=self.y_true, y_pred=self.y_pred, print_cm=True)
        get_precision_recall(y_true=self.y_true, y_pred=self.y_pred)

    def visualize_validation_prediction(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5),
                                             nrows=self.nrows,
                                             ncols=self.ncols,
                                             sharex='all',
                                             sharey='all')

            n_samples = self.validation_data[0].shape[0]

            self.indices = np.random.choice(np.arange(n_samples),
                                            size=self.n_choices,
                                            replace=False)

            x = self.validation_data[0][[self.indices]]

            for i, ax in enumerate(self.ax.flatten()):
                ax.clear()
                ax.imshow(x[i])

            plt.show()

        y_true = self.y_true[self.indices]
        y_pred = self.y_pred[self.indices]

        # check to see if masks, or labels
        try:
            n_imgs, img_height, img_width, img_channel = y_true.shape
            masks = np.concatenate(y_pred, y_true)
            labels = None
        except ValueError:
            n_imgs, n_classes = y_true.shape
            labels = (y_pred, y_true)
            masks = None

        for i, ax in enumerate(self.ax.flatten()):

            if masks is not None:
                if len(masks.shape) == 2:
                    masks = np.expand_dims(masks, axis=2)

                for j in range(masks.shape[2]):
                    mask = masks[:, :, j]
                    if mask.max() > 0:
                        ax.contour(mask, [127.5, ],
                                   colors=self.mask_colors[j])

            if labels is not None:
                y_pred_i = labels[0][i].argmax()
                y_true_i = labels[1][i].argmax()
                ax.set_title('%s/%s' % (y_pred_i, y_true_i))
                if y_pred_i != y_true_i:
                    color = 'red' if y_true_i == 0 else 'magenta'
                else:
                    color = 'green'

                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2.0)
                    ax.spines[axis].set_color(color)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

        # plt.subplots_adjust(wspace=0, hspace=0)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(3)



def config_cls_callbacks(run_name=None):
    callbacks = [
        ValidationPrediction(show_confusion_matrix=True),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.25,
                          patience=2,
                          verbose=1,
                          mode='auto',
                          min_lr=1e-7)
    ]
    if run_name:
        callbacks.extend([
            ModelCheckpoint(get_weights_filename(run_name),
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=True),
            CSVLogger(filename=get_csv_filename(run_name))
        ])
    return callbacks

def backbone(backbone_name, **kwargs):
    """
    Returns a backbone object for the given backbone.
    """
    if 'vgg' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'unet' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'inception' in backbone_name:
        from .inception import InceptionBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    elif 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))

    return b(backbone_name, **kwargs)

# 计算不平衡数据的类权重
from sklearn.utils import compute_class_weight as sk_compute_class_weight
def compute_class_weights(y, wt_type='balanced', return_dict=True):
    # need to check if y is one hot
    if len(y.shape) > 1:
        y = y.argmax(axis=-1)

    assert wt_type in ['ones', 'balanced', 'balanced-sqrt'], 'Weight type not supported'

    classes = np.unique(y)
    class_weights = np.ones(shape=classes.shape[0])

    if wt_type == 'balanced' or wt_type == 'balanced-sqrt':

        class_weights = sk_compute_class_weight(class_weight='balanced',
                                                classes=classes,
                                                y=y)
        if wt_type == 'balanced-sqrt':
            class_weights = np.sqrt(class_weights)

    if return_dict:
        class_weights = dict([(i, w) for i, w in enumerate(class_weights)])

    return class_weights


# 训练
if __name__ == '__main__':

    import sys
#     from models import backbone
    from keras.preprocessing.image import ImageDataGenerator
# 选择一种网络
    # backbone_name = 'vgg16'
    backbone_name = 'resnet50'
    # backbone_name = 'densenet169'
    # backbone_name = 'inception_v3'

# 网络相关参数
    backbone_options = {}
    num_dense_layers = 1
    num_dense_units = 128
    pooling = 'avg'
    dropout_rate = 0.

# 训练相关参数
    dense_layer_regularizer = 'L1'
    class_wt_type = 'ones'
    lr = 1e-4
# k折训练
    num_folds = 5
    for k_fold in range(num_folds):

        version = '0'

        run_name = 'task3_' + backbone_name + '_k' + str(k_fold) + '_v' + version
        # Set prev_run_name to continue training from a previous run
        prev_run_name = None

# 开始训练，加载npy文件
        (x_train, y_train), (x_valid, y_valid), _ = load_training_data(task_idx=3,
                                                                       output_size=224,
                                                                       num_partitions=num_folds,
                                                                       idx_partition=k_fold)
# 可以将bug可视化，便于调bug
        num_classes = y_train.shape[1]
# 回调函数
        callbacks = config_cls_callbacks(run_name)

        from keras.applications import resnet50
        from keras.layers import *
        from keras.models import *
        from keras.optimizers import Adam

        inputs = Input(shape=x_train.shape[1:])
        inputs = Lambda(lambda x: resnet50.preprocess_input(x))(inputs)
        base_model = resnet50.ResNet50(input_tensor=inputs,
                                         include_top=False,
                                         weights='imagenet')
        img_input = base_model.output
        kernel_regularizer = regularizers.l1(1e-4)
        outputs = GlobalAveragePooling2D(name='avg_pool_our')(img_input)
        outputs = Dense(128,
                        activation='relu',
                        name='fc1',
                        kernel_regularizer=kernel_regularizer
                        )(outputs)
        outputs = Dropout(rate=dropout_rate)(outputs)
        outputs = Dense(7,
                        name='predictions',
                        kernel_regularizer=kernel_regularizer
                        )(outputs)
        outputs = Activation('softmax', name='outputs')(outputs)
        model = Model(inputs=base_model.inputs, outputs=outputs)
        model.summary()
        model.compile(optimizer=Adam(lr=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        n_samples_train = x_train.shape[0]
        n_samples_valid = x_valid.shape[0]

        # class_weights = compute_class_weights(y_train, wt_type=class_wt_type)

        batch_size = 4
        use_data_aug = True
        horizontal_flip = True
        vertical_flip = True
        rotation_angle = 180
        width_shift_range = 0.1
        height_shift_range = 0.1

        if use_data_aug:

            datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                         horizontal_flip=horizontal_flip,
                                         vertical_flip=vertical_flip,
                                         width_shift_range=width_shift_range,
                                         height_shift_range=height_shift_range)

            model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=x_train.shape[0] // batch_size * 2,
                                epochs=12,
                                initial_epoch=0,
                                verbose=1,
                                validation_data=(x_valid, y_valid),
                                callbacks=callbacks,
                                workers=8,
                                use_multiprocessing=True)

        else:

            model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=50,
                      verbose=1,
                      validation_data=(x_valid, y_valid),
                      class_weight=class_weights,
                      shuffle=True,
                      callbacks=callbacks)
