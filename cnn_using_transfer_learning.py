from keras.layers import Input,Lambda,Flatten,Dense,Conv2D,MaxPooling2D,GlobalAveragePooling2D
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.datasets import load_files       
from keras.utils import np_utils
import cv2

# define function to load train, test, and validation datasets
def load_dataset(path):
    """

    Parameters
    ----------
    path : path directory
        this is the path which contains the 133 folders of each classes

    Returns
    -------
    images : np array - [num_ex,224,224,3]
        normalized numpy array of each of the images
    dog_targets : TYPE
        DESCRIPTION.

    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    images = []
    tick = 0
    
    for path in dog_files:
        tick += 1
        print(tick)
        image = cv2.imread(path)
        image = cv2.resize(image,(224,224),interpolation = cv2.INTER_LINEAR)
        images.append(image)
    
    #normalizing the images array
    images = np.array(images)
    images = images.astype(np.float32) / 255.0
    
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    
    return images,dog_targets

# load train, test, and validation datasets
# train file contains the path for all the inages
train_images, train_targets = load_dataset(r'E:\programming\dataset\dogImages\train')
valid_images, valid_targets = load_dataset(r'E:\programming\dataset\dogImages\test')
test_images, test_targets = load_dataset(r'E:\programming\dataset\dogImages\valid')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob(r'E:/programming/dataset/dogImages/test/*/'))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('THe shape of training images are {}'.format(train_images.shape))
print('THe shape of valid images are {}'.format(valid_images.shape))
print('THe shape of test images are {}'.format(test_images.shape))



def show_images(index):
    plt.figure(figsize = (8,14))
    for i in range(8):
        plt.subplot(4,2,i+1)
        plt.imshow(train_images[i+index])
        label = np.argmax(train_targets[i+index])
        plt.title(label)
    plt.show()

index = 5 
show_images(index)   



#training a cnn from scratch

model = Sequential()
model.add(Conv2D(filters = 16,kernel_size = (2,2),input_shape = (224,224,3),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 32,kernel_size = (2,2),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 64 ,kernel_size = (2,2),activation = 'relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(units = 133,activation = 'softmax'))

model.summary()


# from keras.callbacks import ModelCheckpoint

epochs = 5

# checkpointer = ModelCheckpoint(filepath = 'saved_models/weights.best.from_scratch.hdf5',
#                                verbose = 1, save_best_only = True)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images,train_targets,validation_data = (valid_images,valid_targets),
          epochs = epochs)
predictions = model.predict_classes(train_images)

from sklearn.metrics import accuracy_score
accuracy_scratch = accuracy_score(y_true = np.argmax(test_targets),y_pred = predictions)
print("The test accuracy is %d" % (accuracy_scratch))


#even after running for 5 epochs the accuracy is still 3-4%
#we will now use transfer learning to train the model

bottleneck_features = np.load(r'E:\programming\trained_models\DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
test_VGG19 = bottleneck_features['test']
valid_VGG19 = bottleneck_features['valid']

print(type(train_VGG19))
train_VGG19.shape

vgg_model = Sequential()
vgg_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
vgg_model.add(Dense(133, activation='softmax'))

vgg_model.summary()
### Define the model

#ompile the model
vgg_model.compile(loss = 'categorical_crossentropy',optimizer = 'rmsprop',metrics = ['accuracy'])


vgg_model.fit(train_VGG19,train_targets,validation_data = (valid_VGG19,valid_targets),epochs = 20,
              batch_size = 20,verbose = 1)

predictions = vgg_model.predict_classes(test_VGG19)
print(predictions.shape)
from sklearn.metrics import accuracy_score,confusion_matrix

test_classes = np.argmax(test_targets,axis = 1)
print(test_classes.shape)
accuracy = accuracy_score(y_true = test_classes,y_pred = predictions)
cm = confusion_matrix(y_true = test_classes,y_pred = predictions)

print('THe test accutacy is {}'.format(accuracy))

# we are achieving a test accuracy of 0.751

vgg_model.save('dog_breed.h5')




