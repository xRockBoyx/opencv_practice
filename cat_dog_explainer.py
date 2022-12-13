from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot    as plt
import tensorflow as tf
import numpy as np
import shap

tf.compat.v1.disable_eager_execution()

# model = Sequential()
# model.add(Convolution2D(32 , 3, 3,  input_shape = (64, 64, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size  = (2,2)))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(2, activation = 'softmax'))

# model.compile(optimizer = 'adam', 
#               loss = 'binary_crossentropy', 
#               metrics = ['accuracy'])

# image_datagen = ImageDataGenerator(rescale = 1./255,
#                                    horizontal_flip=True,
#                                    rotation_range=20,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2, 
#                                    validation_split = 0.2)

# train_set = image_datagen.flow_from_directory(  'shap_Image/',
#                                                  shuffle = True,
#                                                  target_size = (224, 224),
#                                                  batch_size  = 16,
#                                                  subset="training",
#                                                  class_mode  = 'categorical'
#                                                  )
# test_set = image_datagen.flow_from_directory('shap_Image/',
#                                              shuffle = True,
#                                             target_size = (224, 224),
#                                             batch_size  = 16,
#                                             subset="validation",
#                                             class_mode  = 'categorical'
#                                             )

model = load_model('cat_dog.h5')
img = image.load_img('shap_image/Dog/9.jpg', target_size=(64, 64))
img = np.expand_dims(img, axis=0) # 轉換通道
img = img/255 # rescale


pred = model.predict(img)
print(pred)
# x, y = test_set.next()

# image = x[0]
# label = y[0]

# print(label)
# print(test_set.classes[0])
# print(test_set.filepaths[0])
# plt.imshow(image)
# model.fit(  train_set,
#             steps_per_epoch = len(train_set),
#             epochs = 1,
#             validation_data = test_set,
#             validation_steps = len(test_set))

# model.save('cat_dog.h5')

