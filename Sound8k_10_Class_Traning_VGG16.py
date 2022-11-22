import cv2, tensorflow, os, csv, shutil, time
import numpy                as np
import matplotlib.pyplot    as plt
from tqdm                                    import tqdm
from LineNotifyController                    import LineNotifier
from tensorflow.keras.applications.vgg16     import VGG16
from tensorflow.keras.models                 import Sequential
from tensorflow.keras.layers                 import Convolution2D
from tensorflow.keras.layers                 import MaxPooling2D
from tensorflow.keras.layers                 import Flatten
from tensorflow.keras.layers                 import Dense
from tensorflow.keras.layers                 import Dropout
from tensorflow.keras.preprocessing.image    import ImageDataGenerator


BUCKET_NAME          = 'ai-training-notifier-bucket'
ACC_IMAGE_FILE_NAME  = 'VGG16_Acc_8_class_STFT_batch16_input_224.png'
LOSS_IMAGE_FILE_NAME = 'VGG16_loss_8_class_STFT_batch16_input_224.png'
H5_WEIGHT_FILE_NAME  = 'VGG16_Sound8k_8_Class_Epoch_100_Batch_16_STFT_input_224.h5'

Notifier = LineNotifier(notifyToken               = '8sINtMZ1MjV2mOnnbIe0j6KTbiWtlfv6ilzgALwfUai',
                        privateApiKeyJsonFilePath = './line-notifier-image-storage-65936edbb18a.json')

#---------------傳送LINE通知----------------------
Notifier.send_message(text = "\nVGG16訓練開始")
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
                                                 target_size = (224, 224),
                                                 batch_size  = 16,
                                                 subset="training",
                                                 class_mode  = 'categorical'
                                                 )
test_set = image_datagen.flow_from_directory('Images/all_set/',
                                             shuffle = True,
                                            target_size = (224, 224),
                                            batch_size  = 16,
                                            subset="validation",
                                            class_mode  = 'categorical'
                                            )

#---------設定訓練網路-----------
model = Sequential()
model.add(Convolution2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Convolution2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1000,activation="relu"))
model.add(Dense(units=8, activation="softmax"))
model.summary()
model.compile(optimizer = 'adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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