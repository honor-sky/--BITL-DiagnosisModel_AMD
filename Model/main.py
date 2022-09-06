
###############CNN model#############
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf

# Preprocess the image into a 4D array using
# keras.preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow import optimizers
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)



# define image size
img_rows=800
img_cols=800


#load image files
#Bring img idx from excel file
'''
img_csv = pd.read_excel('E:/단국대학교/연구실/[IITP]빅데이터안과진단기술_5년차/dataset/Clean_data/index_220326.xlsx')
img_csv = pd.DataFrame(img_csv)
img_idx = img_csv.loc[:,['file_name','label']]
img_idx = img_idx.iloc[0:2000,:]

## Split train/test data
train_data, test_data = train_test_split(img_idx, test_size=0.3)
print(len(train_data))
print(len(test_data))

# load train images
#X_train = np.zeros(shape=(len(train_data),img_rows,img_cols,3))
#y_train = np.zeros(shape=len(train_data))


## Add train/test data in each folder

img_dir = 'E:/cropped_Image/'

for idx in range(len(train_data)):
    img_path = os.path.join(img_dir, train_data.iloc[idx,0])
    print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #img = image.load_img(img_path, target_size=(800, 800))
    #img_array = image.img_to_array(img)
    #img_array = np.expand_dims(img_array,axis=0)
    if(train_data.iloc[idx,1]==0): #정상
        cv2.imwrite('E:/train/normal/' + train_data.iloc[idx, 0], img)
    else:
        cv2.imwrite('E:/train/md/' + train_data.iloc[idx, 0], img)

    #X_train[idx] = img_path
    #y_train[idx] = train_data.iloc[idx, 1]


# load test images
#X_test = np.zeros(shape=(len(test_data),img_rows,img_cols,3))
#y_test= np.zeros(shape=len(test_data))

for idx in range(len(test_data)):
    img_path = os.path.join(img_dir, test_data.iloc[idx,0])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #img = image.load_img(img_path, target_size=(800, 800))
    #img_array = image.img_to_array(img)
    #img_array = np.expand_dims(img_array,axis=0)
    if (test_data.iloc[idx, 1] == 0):  # 정상
        cv2.imwrite('E:/test/normal/' + test_data.iloc[idx, 0], img)
    else:
        cv2.imwrite('E:/test/md/' + test_data.iloc[idx, 0], img)

    #X_test[idx] = img_path
    #y_test[idx] = test_data.iloc[idx,1]

#print(X_test)
#print(y_test)
'''

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'E:/train',
        target_size=(800,800),
        batch_size=40,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'E:/test',
        target_size=(800,800),
        batch_size=40,
        class_mode='categorical')

print(train_generator.class_indices)

# fix random seed for reproducibility
seed = 100
np.random.seed(seed)
num_classes = 2

# create CNN model
def cnn_model():
    # define model

    model = Sequential()


    model.add(Convolution2D(16, kernel_size=(3,3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(Convolution2D(16, kernel_size=(3,3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(8, kernel_size=(3,3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(Convolution2D(8, kernel_size=(3,3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(4, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(Convolution2D(4, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(2, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(Convolution2D(2, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_rows, img_cols, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))



    # 4~5단계 정도 반복
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model


# build the model
model = cnn_model()
print(model.summary())
# Fit the model
#with tf.device('/gpu:0'):
disp = model.fit(
    train_generator,
    epochs=50,  # 300
    validation_data=test_generator,
    validation_steps=5)

# Final evaluation of the model
scores = model.evaluate(test_generator, steps=5)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])


# summarize history for accuracy
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
plt.savefig('C:/Users/DKU/Desktop/황반변성_CNN img/5000set_2.png')