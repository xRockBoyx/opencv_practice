import cv2, tensorflow, os, csv, shutil, time
import numpy                as np
import matplotlib.pyplot    as plt
from tqdm                                    import tqdm
from LineNotifyController                    import LineNotifier
from tensorflow.keras.models                 import Sequential, Model
from tensorflow.keras.layers                 import Convolution2D
from tensorflow.keras.layers                 import MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers                 import add, Flatten
from tensorflow.keras.layers                 import Input, Dense
from tensorflow.keras.layers                 import Dropout
from tensorflow.keras.layers                 import BatchNormalization
from tensorflow.keras.preprocessing.image    import ImageDataGenerator


BUCKET_NAME          = 'ai-training-notifier-bucket'
ACC_IMAGE_FILE_NAME  = 'Resnet34_Acc_8_class_STFT_batch16.png'
LOSS_IMAGE_FILE_NAME = 'Resnet34_loss_8_class_STFT_batch16.png'
H5_WEIGHT_FILE_NAME  = 'Resnet34_Sound8k_8_Class_Epoch_100_Batch_16_STFT.h5'

Notifier = LineNotifier(notifyToken               = '8sINtMZ1MjV2mOnnbIe0j6KTbiWtlfv6ilzgALwfUai',
                        privateApiKeyJsonFilePath = './line-notifier-image-storage-65936edbb18a.json')

#---------------傳送LINE通知----------------------
Notifier.send_message(text = "\nResnet34訓練開始")
#------------------------------------------------

image_datagen = ImageDataGenerator(rescale = 1./255,
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   validation_split = 0.2)
# test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = image_datagen.flow_from_directory(  'Images/all_set/',
                                                 shuffle = True,
                                                 target_size = (256, 256),
                                                 batch_size  = 16,
                                                 subset="training",
                                                 class_mode  = 'categorical'
                                                 )
test_set = image_datagen.flow_from_directory('Images/all_set/',
                                             shuffle = True,
                                            target_size = (256, 256),
                                            batch_size  = 16,
                                            subset="validation",
                                            class_mode  = 'categorical'
                                            )

#---------設定訓練網路-----------

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = Convolution2D(nb_filter,
                      kernel_size,
                      padding=padding,
                      strides=strides,
                      activation='relu',
                      name=conv_name)(x)
    x = BatchNormalization(axis = 3,name = bn_name)(x)

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x

inpt = Input(shape=(224,224,3))
x = ZeroPadding2D((3,3))(inpt)
x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
#(56,56,64)
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
#(28,28,128)
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
#(14,14,256)
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#(7,7,512)
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
x = AveragePooling2D(pool_size=(7,7))(x)
x = Flatten()(x)
x = Dense(1000,activation='softmax')(x)
x = Dense(8,activation='softmax')(x)

model = Model(inputs=inpt,outputs=x)
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

history = model.fit(train_set,
                    steps_per_epoch  = len(train_set),
                    validation_data  = test_set,
                    epochs           = 100,
                    validation_steps = len(test_set))

model.save(H5_WEIGHT_FILE_NAME)
#-------------------------------

#----------輸出loss圖表-----------------
plt.plot(history.history['loss'])
plt.plot(history.history["val_loss"])
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig(LOSS_IMAGE_FILE_NAME)
plt.show()
#------------------------------------
plt.clf()
#--------------輸出accuracy圖表-------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(ACC_IMAGE_FILE_NAME)
plt.show()
#-----------------------------------

#----------傳送Line通知--------------

Notifier.upload_to_google_bucket(bucketName = BUCKET_NAME,
                                 bucketFileName = ACC_IMAGE_FILE_NAME,
                                 filePath = ACC_IMAGE_FILE_NAME)
Notifier.send_image(text = "\nTrain accuracy : " + 
                            str(history.history['accuracy'][-1]) +
                            "\nVal accuracy : " +
                            str(history.history['val_accuracy'][-1]),
                    bucketFileName = ACC_IMAGE_FILE_NAME)

Notifier.upload_to_google_bucket(bucketName = BUCKET_NAME,
                                 bucketFileName = LOSS_IMAGE_FILE_NAME,
                                 filePath = LOSS_IMAGE_FILE_NAME)
Notifier.send_image(text = "\nTrain loss : " +
                            str(history.history['loss'][-1]) +
                            "\nVal loss : " +
                            str(history.history['val_loss'][-1]),
                    bucketFileName = LOSS_IMAGE_FILE_NAME)
#-------------------------------------