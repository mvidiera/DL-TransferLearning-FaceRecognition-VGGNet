#Transfer Learning: 

#THIS IS THE GENERIC TEMPLATE FOR CNN. 

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model

from keras.applications.vgg16 import VGG16 # VGG gets 1000 different category in last vector and softmax is applied at last

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
#image augmentation is done by ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
#to create seq model as vgg16
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this. VGG ip image is always 244, 244

IMAGE_SIZE = [224, 224]

train_path = '/Users/melissavidiera/Documents/Deep Learning/Transfer-Learning-master/FaceRecognition/Train'
valid_path = '/Users/melissavidiera/Documents/Deep Learning/Transfer-Learning-master/FaceRecognition/Test'

# add preprocessing layer to the front of VGG
# in this pre prcoessing layer of VGG,for inout_shape I pass 224x244 and + [3] which is channels rgb 
#include_top= false: we are telling whether we are adding last layer in vgg are not. false here means not to include. used to transfer leaning
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# don't train existing weights
#this is very important not to train
# in vgg weights are already fixed, that's why we use for loop for all the layers and make trainable as false
# if we dont do it model will re train numberous time which takes up lot of resources that presently I dont have and accuracy will be less

for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
  #after executing I can check in variable explorer that I have 3 folders inside
folders = glob('/Users/melissavidiera/Documents/Deep Learning/Transfer-Learning-master/FaceRecognition/Train/*')
  
#flattening last layer
# our layers - you can add more if you want
x = Flatten()(vgg.output)

# x = Dense(1000, activation='relu')(x)
# appending my folders as dense layer with activation func softmax. number of folders= number of cateories(in this case 3) 
#softmax will be applied for these 3
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()
# in output of this line, I can see that last layer has 3 categories(dense_1) that is Ranbir, Ranvir and Manav

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#image augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
                                                  
training_set = train_datagen.flow_from_directory('/Users/melissavidiera/Documents/Deep Learning/Transfer-Learning-master/FaceRecognition/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/Users/melissavidiera/Documents/Deep Learning/Transfer-Learning-master/FaceRecognition/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_new_model.h5')

