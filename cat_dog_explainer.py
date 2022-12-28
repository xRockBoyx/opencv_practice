# this is the code from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import cv2
import numpy as np
import shap
import json

# tf.compat.v1.disable_v2_behavior()

# url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
# with open(shap.datasets.cache(url)) as file:
#     class_names = [v[1] for v in json.load(file).values()]
# #print("Number of ImageNet classes:", len(class_names))
# print("Class names:", class_names)


model = load_model('Resnet50v2_Cat_Dog_Epoch_100_Batch_16.h5')
class_names = ['cat', 'dog']
image = cv2.imread("archive/test_set/dogs/dog.4011.jpg")
image1 = cv2.imread("archive/test_set/dogs/dog.4001.jpg")
image2 = cv2.imread("archive/test_set/dogs/dog.4002.jpg")
# print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#cv讀照片，顏色莫認為BGR，需轉為RGB，錯誤表示黑白或已轉
image = cv2.resize(image, (224, 224))
image1 =  cv2.resize(image1, (224, 224))
image2 =  cv2.resize(image2, (224, 224))
image = np.expand_dims(image, axis = 0)
image1 = np.expand_dims(image1, axis = 0)
image2 = np.expand_dims(image2, axis = 0)

n = np.concatenate([image, image1])
n = np.concatenate([n, image2])
print(n.shape)
print(n[2].shape)
n  = n / 255.0

masker = shap.maskers.Image("inpaint_telea", n[0].shape)
explainer = shap.Explainer(model, masker, output_names = class_names)


shap_values = explainer(n[0:1], 
                        max_evals=200, 
                        batch_size=30,
                        outputs=shap.Explanation.argsort.flip[:2])
# print(shap_values)
print(np.argmax(model.predict(n), axis=1))
print(model.predict(n))
shap.image_plot(shap_values)
# shap.image_plot(shap_values)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# ...include code from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py