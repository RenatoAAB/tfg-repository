import os
import numpy as np
import glob
import keras
import zipfile

from keras_video import VideoFrameGenerator

import tensorflow as tf
from tensorflow.keras import initializers, layers, models, optimizers, metrics, regularizers
from keras import backend as K 
from kerastuner.tuners import RandomSearch
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, 
    TimeDistributed, Dense, Dropout, LSTM
    
# Global params
CLASSES = 2
SIZE = (64, 64)
CHANNELS = 3
NBFRAME = 20
BS = 32
SHAPE = SIZE + (CHANNELS,)
INSHAPE=(NBFRAME,) + SHAPE


def build_convnet(hp):
    #momentum = .9
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=SHAPE))

    for i in range(hp.Int('conv_blocks', 3, 5, default=3)):
        filters = hp.Int('filters_' + str(i), 32, 256, step=32)
        for _ in range(2):
            model.add(Conv2D(filters, (3,3), padding='same'))
            model.add(BatchNormalization())
            model.add(ReLU())

        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            model.add(MaxPooling2D())
        else:
            model.add(AveragePooling2D())

    model.add(GlobalAvgPool2D())

    return model

def action_model(hp):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(hp)
    
    # then create our final model
    model = keras.Sequential()
    # add the convnet with (NBFRAME, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=INSHAPE))
    
    # LSTM
    model.add(LSTM(
        hp.Int('LSTM', 32, 128, step=32))
    )
    # and finally, we make a decision network
    for i in range(hp.Int('dense_blocks', 2, 4, default=2)):
        filters = hp.Int('filters_' + str(i), 32, 512, step=32)
        model.add(Dense(filters, activation='relu', kernel_regularizer=keras.regularizers.l2(
            hp.Choice('l2_'+ str(i), values=[1e-2, 1e-3]))
        ))
        model.add(Dropout(hp.Float('dropout_' + str(i), 0, 0.7, step=0.1, default=0.5)))

    model.add(Dense(CLASSES, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        metrics=['accuracy']
    )

    return model

def if __name__ == "__main__":

    local_zip = '/content/tfg-repository/mitosis_images_test.zip'

    zip_ref = zipfile.ZipFile(local_zip, 'r')

    zip_ref.extractall('/content/tmp')
    zip_ref.close()

    # Pattern to get videos and classes
    glob_pattern='/content/tmp/{classname}/*'

    COUNT_NORMAL = len(os.listdir('/content/tmp/NormalMitosis'))
    COUNT_ABNORMAL = len(os.listdir('/content/tmp/AbnormalMitosis'))

    # for data augmentation
    data_aug = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=90
    )

    # Create video frame generator
    train = VideoFrameGenerator(
        rescale=1/255.,
        glob_pattern=glob_pattern,
        nb_frames=NBFRAME,
        split_val=.2, 
        split_test=.25,
        shuffle=True,
        batch_size=BS,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=False
    )

    valid = train.get_validation_generator()

    test = train.get_test_generator()

    # Class bias
    initial_bias = np.log([COUNT_NORMAL / COUNT_ABNORMAL])

    weight_for_0 = (1 / COUNT_NORMAL) * (len(train)) / 2.0
    weight_for_1 = (1 / COUNT_ABNORMAL) * (len(train)) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    # Perform tuning
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='tune_dir',
        project_name='tune_example'
    )

    # Display search space summary
    tuner.search_space_summary()

    # Perform random search
    tuner.search(
        train,
        validation_data=valid,
        epochs=10,
        class_weight=class_weight
    )

    hist = model.fit(
        train,
        validation_data=valid,
        epochs=EPOCHS,
        verbose=1,
        class_weight=class_weight,
        #callbacks=[early_stop]
    )