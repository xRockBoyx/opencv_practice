import cv2, tensorflow, os, csv, shutil, time
import numpy                as np
import matplotlib.pyplot    as plt
from tqdm                                   import tqdm
from LineNotifyController                   import LineNotifier
from tensorflow.keras.models                import Sequential
from tensorflow.keras.layers                import Convolution2D
from tensorflow.keras.layers                import MaxPooling2D
from tensorflow.keras.layers                import Flatten
from tensorflow.keras.layers                import Dense
from tensorflow.keras.layers                import Dropout
from tensorflow.keras.preprocessing.image   import ImageDataGenerator

BUCKET_NAME          = 'ai-training-notifier-bucket'
ACC_IMAGE_FILE_NAME  = 'Acc_8_class_STFT_Epoch_100_batch16_filter128.png'
LOSS_IMAGE_FILE_NAME = 'loss_8_class_STFT_Epoch_100_batch16_filter128.png'

Notifier = LineNotifier(notifyToken               = '8sINtMZ1MjV2mOnnbIe0j6KTbiWtlfv6ilzgALwfUai',
                        privateApiKeyJsonFilePath = './line-notifier-image-storage-65936edbb18a.json')

#---------------傳送LINE通知----------------------
Notifier.send_message(text = "\nCNN訓練開始")
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

# This part we train our model without ImageDataGenerator(不做資料擴增的意思)
# class_names = ['dog_bark', 
#                'children_playing',
#                'car_horn', 
#                'air_conditioner', 
#                'engine_idling', 
#                'siren',
#                'jackhammer',
#                'drilling',
#                'street_music',
#                'gun_shot']
# class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
# nb_classes = len(class_names)

# (train_images, train_labels), (test_images, test_labels) = load_data()
# train_images = train_images / 255.0 
# test_images = test_images / 255.0

#---------設定訓練網路-----------
model = Sequential()
model.add(Convolution2D(256 , 3, 3, padding = 'same', input_shape = (256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size  = (2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(512 , 3, 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size  = (2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(8, activation = 'softmax'))
#-------------------------------
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
      
history = model.fit(    train_set,
                        steps_per_epoch = len(train_set),
                        epochs = 100,
                        validation_data = test_set,
                        validation_steps = len(test_set)
                        )
# history = model.fit(train_images, train_labels, 
#                     validation_data=(test_images, test_labels),
#                     #verbose=2,callbacks=[earlyStop],
#                     verbose = 2,
#                     batch_size=32, 
#                     epochs=100)
model.save('Sound8k_8_Class_Epoch_100_Batch16_STFT_Filter128.h5')

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