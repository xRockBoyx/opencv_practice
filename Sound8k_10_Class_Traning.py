import cv2, tensorflow, os, csv, shutil
import numpy                as np
import matplotlib.pyplot    as plt
from tqdm                    import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def load_data():
    datasets = ['Images/training_set', 'Images/test_set']#資料夾
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            if folder == ".DS_Store":
                continue
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                if file == "desktop.ini":
                    break
                if file == ".DS_Store":
                    continue
                # Open and resize the img
                # print(img_path)
                image = cv2.imread(img_path)
                # print(image.shape)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #cv讀照片，顏色莫認為BGR，需轉為RGB，錯誤表示黑白或已轉
                image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output

# This part we train our model without ImageDataGenerator(不做資料擴增的意思)
class_names = ['dog_bark', 
               'children_playing',
               'car_horn', 
               'air_conditioner', 
               'engine_idling', 
               'siren',
               'jackhammer',
               'drilling',
               'street_music',
               'gun_shot']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)
IMAGE_SIZE = (256, 256)

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images / 255.0 
test_images = test_images / 255.0

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
model.add(Dense(10, activation = 'softmax'))
#-------------------------------
model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])

history = model.fit(train_images, train_labels, 
                    validation_data=(test_images, test_labels),
                    #verbose=2,callbacks=[earlyStop],
                    verbose = 2,
                    batch_size=8, 
                    epochs=100)
model.save('Sound8k_10_Class_Epoch_100_Batch_32.h5')

#----------輸出loss圖表-----------------
plt.plot(history.history['loss'])
plt.plot(history.history["val_loss"])
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('loss.png')
plt.show()
#------------------------------------

#--------------輸出accuracy圖表-------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('Acc.png')
plt.show()
#-----------------------------------