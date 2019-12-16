# -*- coding: utf-8 -*-
#打印，输出
import numpy as np
import sys
import logging
import sys
import os
import inspect
import pandas as pd
import keras
from matplotlib import pyplot as plt
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.utils.vis_utils import plot_model
from keras.applications import vgg16
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import conv_utils
from keras.engine.topology import get_source_inputs

plt.ion()

def _rot90(images):
    return K.permute_dimensions(K.reverse(images, [2]), [0, 2, 1, 3])


def _rot180(images):
    return K.reverse(images, [1, 2])


def _rot270(images):
    return K.reverse(K.permute_dimensions(images, [0, 2, 1, 3]), [2])


def rot90_4D(images, k):
    def _rot90():
        return array_ops.transpose(array_ops.reverse_v2(images, [2]), [0, 2, 1, 3])

    def _rot180():
        return array_ops.reverse_v2(images, [1, 2])

    def _rot270():
        return array_ops.reverse_v2(array_ops.transpose(images, [0, 2, 1, 3]), [2])

    cases = [(math_ops.equal(k, 1), _rot90),
             (math_ops.equal(k, 2), _rot180),
             (math_ops.equal(k, 3), _rot270)]

    result = control_flow_ops.case(
        cases, default=lambda: images, exclusive=True)

    shape = result.get_shape()
    result.set_shape([shape[0], None, None, shape[3]])
    return result

def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)



root_dir = '/home/deeplearning/wwt/weight_test/'

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

task1_gt = 'ISIC2018_Task1_Training_GroundTruth'

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


task1_gt_dir = os.path.join(data_dir, task1_gt)


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

task12_images_npy_prefix = 'task12_images'
task12_validation_images_npy_prefix = 'task12_validation_images'
task12_test_images_npy_prefix = 'task12_test_images'

task1_gt_npy_prefix = 'task1_masks'


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

#分界线


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

def config_seg_callbacks(run_name=None):
    callbacks = [
        ValidationPrediction(show_confusion_matrix=False),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=2,
                          verbose=1,
                          mode='auto',
                          min_lr=1e-7),
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

def load_task12_training_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task12_images_npy_prefix, suffix))
    
    images = np.load(images_npy_filename)
    return images

def load_task1_training_masks(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    npy_filename = os.path.join(cached_data_dir, 'task1_masks%s.npy' % suffix)
    masks = np.load(npy_filename)
    return masks

def load_training_data(task_idx,
                       output_size=None,
                       num_partitions=5,
                       idx_partition=0,
                       test_split=0.):
    x = load_task12_training_images(output_size=output_size)
    y = load_task1_training_masks(output_size=output_size)
    return partition_data(x=x, y=y, k=num_partitions, i=idx_partition, test_split=test_split)

def partition_data(x, y, k=5, i=0, test_split=1. / 6, seed=42):
    assert isinstance(k, int) and isinstance(i, int) and 0 <= i < k

    n = x.shape[0]

    n_set = int(n * (1. - test_split)) // k
    # divide the data into (k + 1) sets, -1 is test set, [0, k) are for train and validation
    indices = np.array([i for i in range(k) for _ in range(n_set)] +
                       [-1] * (n - n_set * k),
                       dtype=np.int8)

    np.random.seed(seed)
    np.random.shuffle(indices)

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

def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None

def __conv_block(nb_filters,
                 activation='relu',
                 block_prefix=None):

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        # 'kernel_initializer': kernel_initializer,
        # 'bias_initializer': bias_initializer,
    }

    nb_layers_per_block = 1 if isinstance(nb_filters, int) else len(nb_filters)
    nb_filters = conv_utils.normalize_tuple(nb_filters, nb_layers_per_block, 'nb_filters')

    def block(x):
        for i, n in enumerate(nb_filters):
            x = Conv2D(filters=nb_filters[i],
                       name=name_or_none(block_prefix, '_conv%d' % (i+1)),
                       **options)(x)

            if activation.lower() == 'leakyrelu':
                x = LeakyReLU(alpha=0.33)(x)
            else:
                x = Activation(activation)(x)
        return x
    return block

def __transition_up_block(nb_filters,
                          merge_size,
                          upsampling_type='deconv',
                          block_prefix=None):
    options = {
        'padding': 'same'
    }
    merge_size = conv_utils.normalize_tuple(merge_size, 2, 'merge_size')

    def block(ip):
        try:
            src, dst = ip
        except TypeError:
            src = ip
            dst = None

        # copy and crop
        if K.image_data_format() == 'channels_last':
            indices = slice(1, 3)
            channel_axis = -1
        else:
            indices = slice(2, 4)
            channel_axis = 1

        src_height, src_width = K.get_variable_shape(src)[indices]

        target_height, target_width = merge_size
        scale_factor = ((target_height + src_height - 1) // src_height,
                        (target_width + src_width - 1) // src_width)

        # upsample and crop
        if upsampling_type == 'upsample':
            x = UpSampling2D(size=scale_factor,
                             name=name_or_none(block_prefix, '_upsampling'))(src)
            x = Conv2D(nb_filters, (2, 2),
                       activation='relu', padding='same', name=name_or_none(block_prefix, '_conv'))(x)
        elif upsampling_type == 'subpixel':
            x = Conv2D(nb_filters, (2, 2),
                       activation='relu', padding='same', name=name_or_none(block_prefix, '_conv'))(src)
            x = SubPixelUpscaling(scale_factor=scale_factor,
                                  name=name_or_none(block_prefix, '_subpixel'))(x)
        else:
            x = Conv2DTranspose(nb_filters, (2, 2), strides=scale_factor,
                                name=name_or_none(block_prefix, '_deconv'),
                                **options)(src)

        if src_height * scale_factor[0] > target_height or src_width * scale_factor[1] > target_width:
            height_padding, width_padding = (src_height - target_height) // 2, (src_width - target_width) // 2
            x = Cropping2D(cropping=(height_padding, width_padding),
                           name=name_or_none(block_prefix, 'crop1'))(x)

        if dst is None:
            return x

        dst_height, dst_width = K.get_variable_shape(dst)[indices]

        # copy and crop
        if dst_height > target_height or dst_width > target_width:
            height_padding, width_padding = ((dst_height - target_height) // 2, (dst_width - target_width) // 2)
            dst = Cropping2D(cropping=(height_padding, width_padding),
                             name=name_or_none(block_prefix, 'crop2'))(dst)

        x = Concatenate(axis=channel_axis, name=name_or_none(block_prefix, '_merge'))([x, dst])

        return x

    return block

def __normalize_target_size(curr_size, target_size, scale_factor):
    while curr_size < target_size:
        target_size //= scale_factor
    return target_size

def save_model_to_run(model, run_name):
    json_path = get_json_filename(run_name)
    h5_path = get_weights_filename(run_name)

    with open(json_path, 'w') as json_file:
        json_file.write(model.to_json())

    model.save_weights(h5_path)

def plot_model(save_to_dir, model, name):
    filename = os.path.join(model_data_dir, save_to_dir, '%s.png' % name)
    keras.utils.plot_model(model,
                           to_file=filename,
                           show_shapes=True,
                           show_layer_names=True)

def pixelwise_precision(num_classes=1):
    def binary_pixelwise_precision(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2, 3])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    def categorical_pixelwise_precision(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    if num_classes == 1:
        return binary_pixelwise_precision
    else:
        return categorical_pixelwise_precision


def pixelwise_recall(num_classes=1):
    return pixelwise_sensitivity(num_classes)


def pixelwise_sensitivity(num_classes=1):
    def binary_pixelwise_sensitivity(y_true, y_pred):
        # indices = tf.where(K.greater_equal(y_true, 0.5))
        # y_pred = tf.gather_nd(y_pred, indices)

        y_true = K.round(y_true)
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total_pos = K.sum(K.abs(y_true), axis=[1, 2, 3])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    def categorical_pixelwise_sensitivity(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total_pos = K.sum(K.abs(y_true), axis=[1, 2])
        return K.mean(true_pos / K.clip(total_pos, K.epsilon(), None), axis=-1)

    if num_classes == 1:
        return binary_pixelwise_sensitivity
    else:
        return categorical_pixelwise_sensitivity


def pixelwise_specificity(num_classes=1):
    def binary_pixelwise_specificity(y_true, y_pred):
        true_neg = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2, 3])
        total_neg = K.sum(K.abs(1. - y_true), axis=[1, 2, 3])
        return true_neg / K.clip(total_neg, K.epsilon(), None)

    def categorical_pixelwise_specificity(y_true, y_pred):
        y_true, y_pred = y_true[..., 1:], y_pred[..., 1:]
        true_neg = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2])
        total_neg = K.sum(K.abs(1. - y_true), axis=[1, 2])
        return true_neg / K.clip(total_neg, K.epsilon(), None)
    if num_classes == 1:
        return binary_pixelwise_specificity
    else:
        return categorical_pixelwise_specificity


def dice_coeff(num_classes=1):
    def binary_dice_coeff(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
        dice = 2 * intersection / K.clip(union, K.epsilon(), None)
        return dice

    def categorical_dice_coeff(y_true, y_pred):

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2])
        dice = 2 * intersection / K.clip(union, K.epsilon(), None)
        return K.mean(dice, axis=-1)

    if num_classes == 1:
        return binary_dice_coeff
    else:
        return categorical_dice_coeff


def class_jaccard_index(idx):
    def jaccard_index(y_true, y_pred):
        y_true, y_pred = y_true[..., idx], y_pred[..., idx]
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        # Adding all three axis to average across images before dividing
        # See https://forum.isic-archive.com/t/task-2-evaluation-and-superpixel-generation/417/2
        intersection = K.sum(K.abs(y_true * y_pred), axis=[0, 1, 2])
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[0, 1, 2])
        jac = intersection / K.clip(sum_ - intersection, K.epsilon(), None)
        return jac
    return jaccard_index
    
def jaccard_index(num_classes):
    def binary_jaccard_index(y_true, y_pred):
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
        iou = intersection / K.clip(union - intersection, K.epsilon(), None)
        return iou

    def categorical_jaccard_index(y_true, y_pred):
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        intersection = K.abs(y_true * y_pred)
        union = K.abs(y_true) + K.abs(y_pred)

        intersection = K.sum(intersection, axis=[0, 1, 2])
        union = K.sum(union, axis=[0, 1, 2])

        iou = intersection / K.clip(union - intersection, K.epsilon(), None)
        # iou = K.mean(iou, axis=-1)
        return iou

    if num_classes == 1:
        return binary_jaccard_index
    else:
        return categorical_jaccard_index   




if __name__ == '__main__':
    task_idx = 1
    version = '0'

    num_folds = 5

    for k_fold in range(num_folds):

        # backbone_name = 'unet'
        # backbone_name = 'inception_v3'
#         backbone_name = 'resnet50'
        # backbone_name = 'densenet169'

        backbone_name = 'vgg16'

        # Network architecture
        upsampling_type = 'deconv'
        bottleneck = True
        batch_normalization = False
        init_nb_filters = 32
        growth_rate = 2
        nb_blocks = 5
        nb_layers_per_block = 2
        max_nb_filters = 512

        encoder_activation = 'relu'
        decoder_activation = 'relu'
        use_activation = True
        use_soft_mask = False

        if backbone_name == 'unet':
            backbone_options = {
                'nb_blocks': nb_blocks,
                'init_nb_filters': init_nb_filters,
                'growth_rate': growth_rate,
                'nb_layers_per_block': nb_layers_per_block,
                'max_nb_filters': max_nb_filters,
                'activation': encoder_activation,
                'batch_normalization': batch_normalization,
            }
        else:
            backbone_options = {}

        # training parameter
        batch_size = 32
        initial_epoch = 0
        epochs = 25
        init_lr = 1e-4  # Note learning rate is very important to get this to train stably
        min_lr = 1e-7
        patience = 1

        # data augmentation parameters
        use_data_aug = True
        horizontal_flip = True
        vertical_flip = True
        rotation_angle = 180
        width_shift_range = 0.1
        height_shift_range = 0.1

        model_name = 'task%d_%s' % (task_idx, backbone_name)
        run_name = 'task%d_%s_k%d_v%s' % (task_idx, backbone_name, k_fold, version)
        from_run_name = None

        debug = False
        print_model_summary = True
        plot_model_summary = True

        (x_train, y_train), (x_valid, y_valid), _ = load_training_data(task_idx=task_idx,
                                                                       output_size=224,
                                                                       num_partitions=num_folds,
                                                                       idx_partition=k_fold)

        # Target should be of the type N x 224 x 224 x 1
        if len(y_train.shape) == 3:

            y_train = y_train[..., None]
            y_valid = y_valid[..., None]

        if y_train[0].max() > 1:
            if use_soft_mask:
                y_train = y_train / 255.
                y_valid = y_valid / 255.
            else:
                y_train = (y_train > 127.5).astype(np.uint8)
                y_valid = (y_valid > 127.5).astype(np.uint8)
        else:
            y_train = y_train.astype(np.uint8)
            y_valid = y_valid.astype(np.uint8)

        n_samples_train = x_train.shape[0]
        n_samples_valid = x_valid.shape[0]

        callbacks = config_seg_callbacks(run_name)

        inputs = Input(shape=x_train.shape[1:])
        indices = slice(0, 2)
        input_shape = (224, 224, 3)
        output_size = input_shape[slice(0, 2)]
        backbone_layer_names = ['block1_conv2',
                                    'block2_conv2',
                                    'block3_conv3',
                                    'block4_conv3',
                                    'block5_conv3']
       
        inputs = Lambda(lambda x: vgg16.preprocess_input(x))(inputs)
        base_model = vgg16.VGG16(input_tensor=inputs,
                                       include_top=False,
                                       weights='imagenet')
        backbone_layers = [base_model.get_layer(name=layer_name) for layer_name in backbone_layer_names]
        backbone_features = [backbone_layer.output for backbone_layer in backbone_layers]

        output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
        output_height, output_width = output_size

        init_nb_filters = 32
        __init_nb_filters = 32
        indices = slice(1, 3)
        channel = 3
        growth_rate = 2
        max_nb_filters = 512
        nb_layers_per_block = 2
        upsampling_type = 'deconv'
        use_activation = True
        save_to = run_name
        num_classes = y_train.shape[3]
        activation= "relu"
        name=model_name
        scale_factor=2
        
        assert task_idx in {1, 2}
        
        if task_idx == 1:
            metrics = [jaccard_index(num_classes),
                       pixelwise_sensitivity(num_classes),
                       pixelwise_specificity(num_classes)]

        features=backbone_features
        nb_features = len(features)
        feature_shapes = [K.get_variable_shape(feature) for feature in features]
        feature_sizes = [feature_shape[indices] for feature_shape in feature_shapes]

        feature_height, feature_width = feature_sizes[0]
        if feature_height < output_height or feature_width < output_width:
            __init_nb_filters = int(__init_nb_filters * growth_rate)

        bottleneck = True
        if bottleneck:
            for i in range(nb_features - 1, -1, -1):
                feature_shape = feature_shapes[i]
                nb_filters = int(__init_nb_filters * (growth_rate ** i))
                nb_filters = min(nb_filters, max_nb_filters)
                if feature_shape[channel] > nb_filters:
                    features[i] = Conv2D(nb_filters, 1,
                                         padding='same',
                                         activation='relu',
                                         name='feature%d_bottleneck' % (i + 1))(features[i])

        nb_layers_per_block = conv_utils.normalize_tuple(nb_layers_per_block, nb_features, 'nb_layers_per_block')

        x = features[-1]

        for i in range(nb_features - 1, 0, -1):
            dst = features[i - 1]
            dst_height, dst_width = feature_sizes[i - 1]

            merge_size = __normalize_target_size(dst_height, output_height, scale_factor)
            if dst_width != dst_height:
                merge_size = (merge_size, __normalize_target_size(dst_width, output_width, scale_factor))

            nb_filters = int(__init_nb_filters * (growth_rate ** (i - 1)))
            nb_filters = min(nb_filters, max_nb_filters)

            x = __transition_up_block(nb_filters=nb_filters,
                                      merge_size=merge_size,
                                      block_prefix='feature%d' % (i + 1),
                                      upsampling_type=upsampling_type)([x, dst])

            x = __conv_block(nb_filters=conv_utils.normalize_tuple(nb_filters,
                                                                   nb_layers_per_block[i - 1],
                                                                   'nb_filters'),
                             activation=activation,
                             block_prefix='feature%d' % i)(x)

        if __init_nb_filters > init_nb_filters:
            x = __transition_up_block(nb_filters=init_nb_filters,
                                      merge_size=output_size,
                                      block_prefix='decoder_block%d' % (nb_features + 1),
                                      upsampling_type=upsampling_type)(x)

            x = __conv_block(nb_filters=[init_nb_filters],
                             activation=activation,
                             block_prefix='feature%d' % (nb_features + 1))(x)

        include_top = True
        if include_top:
            x = Conv2D(num_classes, (1, 1), activation='linear', name='predictions')(x)
            if use_activation:
                output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
                x = Activation(output_activation, name='outputs')(x)

        outputs = x
        model = Model(inputs=base_model.inputs, outputs=outputs, name=name)

        model.summary()
        if plot_model_summary and save_to and not on_aws():
            plot_model(save_to_dir=save_to, model=model, name=name)
        save_model_to_run(model, save_to)
        model.compile(optimizer=Adam(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=metrics)

        if use_data_aug:

            data_gen_args = dict(horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 rotation_range=rotation_angle,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range)

            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)

            seed = 42

            image_datagen.fit(x_train, augment=True, seed=seed)
            mask_datagen.fit(y_train, augment=True, seed=seed)

            image_generator = image_datagen.flow(x=x_train, batch_size=batch_size, seed=seed)
            mask_generator = mask_datagen.flow(x=y_train, batch_size=batch_size, seed=seed)

            train_generator = zip(image_generator, mask_generator)

            model.fit_generator(generator=train_generator,
                                steps_per_epoch=n_samples_train // batch_size,
                                epochs=epochs,
                                initial_epoch=initial_epoch,
                                verbose=1,
                                validation_data=(x_valid, y_valid),
                                callbacks=callbacks,
                                workers=8,
                                use_multiprocessing=False)
        else:

            model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_valid, y_valid),
                      shuffle=True,
                      callbacks=callbacks)
