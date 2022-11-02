from tabnanny import verbose
import cv2, tensorflow, os, csv, shutil
import numpy                as np
import matplotlib.pyplot    as plt
from tqdm                                   import tqdm
from tensorflow.keras.models                import Sequential
from tensorflow.keras.layers                import Convolution2D
from tensorflow.keras.layers                import MaxPooling2D
from tensorflow.keras.layers                import Flatten
from tensorflow.keras.layers                import Dense
from tensorflow.keras.layers                import Dropout
from tensorflow.keras.preprocessing.image   import ImageDataGenerator

# gpus = tensorflow.config.experimental.list_physical_devices('GPU')

# for gpu in gpus:
#     tensorflow.config.experimental.set_memory_growth(gpu, True)

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('Images/training_set/',
                                                 target_size = (256, 256),
                                                 batch_size  = 32,
                                                 class_mode  = 'categorical'
                                                 )
test_set = test_datagen.flow_from_directory('Images/test_set/',
                                            target_size = (256, 256),
                                            batch_size  = 32,
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
model.add(Convolution2D(128 , 3, 3, padding = 'same', input_shape = (256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size  = (2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(256 , 3, 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size  = (2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
#-------------------------------
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
      
history = model.fit(    train_set,
                        steps_per_epoch = len(train_set),
                        epochs = 100,
                        validation_data = test_set,
                        validation_steps = len(test_set),
                        verbose = 2)
# history = model.fit(train_images, train_labels, 
#                     validation_data=(test_images, test_labels),
#                     #verbose=2,callbacks=[earlyStop],
#                     verbose = 2,
#                     batch_size=32, 
#                     epochs=100)
model.save('Sound8k_10_Class_Epoch_100_Batch_32_STFT.h5')

#----------輸出loss圖表-----------------
plt.plot(history.history['loss'])
plt.plot(history.history["val_loss"])
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('loss_STFT_no_argumentation.png')
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
plt.savefig('Acc_STFT_no_argumentation.png')
plt.show()
#-----------------------------------