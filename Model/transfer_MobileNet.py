import numpy as np
import pandas as pd
import os
import cv2
import random
import tensorflow as tf
from PIL import Image
import PIL.ImageOps
from os import remove

# keras.preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
import keras.engine
from keras.layers import Dense,GlobalMaxPooling2D
from keras import Model
from tensorflow import optimizers
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

## 모든 이미지 사용
'''
img_csv = pd.read_excel('E:/단국대학교/연구실/[IITP]빅데이터안과진단기술_5년차/dataset/Clean_data/index_220326.xlsx')
img_csv = pd.DataFrame(img_csv)
img_idx = img_csv.loc[:,['file_name','label']]
img_idx = img_idx.iloc[0:5000,:]
print(len(img_idx))


## Split train/test data
train_data, test_data = train_test_split(img_idx, test_size=0.4)
print(len(train_data))
print(len(test_data))


## Add train/test data in each folder
img_dir = 'E:/cropped_Image/'   ## BAI 데이터도 포함
for idx in range(len(train_data)):
    img_path = os.path.join(img_dir, train_data.iloc[idx,0])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if(train_data.iloc[idx,1]==0): #정상
        cv2.imwrite('E:/AMD_dataset/train/normal/' + train_data.iloc[idx, 0], img)
    else:
        cv2.imwrite('E:/AMD_dataset/train/md/' + train_data.iloc[idx, 0], img)

for idx in range(len(test_data)):
    img_path = os.path.join(img_dir, test_data.iloc[idx,0])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if (test_data.iloc[idx, 1] == 0):  # 정상
        cv2.imwrite('E:/AMD_dataset/test/normal/' + test_data.iloc[idx, 0], img)
    else:
        cv2.imwrite('E:/AMD_dataset/test/md/' + test_data.iloc[idx, 0], img)
'''

'''
##### 데이터 증강 #####
file_path = 'D:/AMD_dataset/test/md/'
file_names = os.listdir(file_path)
total_origin_image_num = len(file_names)
augment_cnt = 1

num_augmented_images = 618  #1600
## cv2 배운거 응용
for i in range(num_augmented_images):
    ## trian의 전체 md 이미지 중에서 랜덤하게 하나를 뽑음
    #change_picture_index = random.randrange(1, total_origin_image_num - 1) # 여기서 어지간하면 안 겹치게....
    #print(change_picture_index)
    #print(file_names[change_picture_index])


    if (random_augment == 1):
        # 이미지 좌우 반전
        #print("invert")
        inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT) #Image.FLIP_LEFT_RIGHT
        inverted_image.save(file_path + file_name +'_invert' + str(augment_cnt) +'.jpg')

  
    elif (random_augment == 2):
        # 이미지 기울이기
        #print("rotate")
        rotated_image = image.rotate(random.randrange(-20, 20))
        rotated_image.save(file_path +file_name+ '_rotated' + str(augment_cnt) +'.jpg')

    elif (random_augment == 3):
        # 노이즈 추가하기
        img = cv2.imread(origin_image_path)
        #print("noise")
        row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_array = img + gauss
        noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
        noisy_image.save(file_path + file_name + '_noiseAdded' + str(augment_cnt) +'.jpg')

    augment_cnt += 1
'''



# define image size
img_rows=800
img_cols=800

n_batch = 40

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'E:/AMD_dataset/train',
        target_size=(800,800),
        batch_size=n_batch, #150
        class_mode='categorical')
## generator에서 증강 옵션 넣기


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'E:/AMD_dataset/test',
        target_size=(800,800),
        batch_size=n_batch, #200
        class_mode='categorical')

# fix random seed for reproducibility
seed = 100
np.random.seed(seed)
num_classes = 2

# create MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', alpha=0.35,include_top=False, input_shape=(img_rows, img_cols, 3),classes=num_classes)
base_model.summary()

# Extract the last layer from third block of model
last = GlobalMaxPooling2D()(base_model.output)
#last = base_model.get_layer('out_relu').output

# Add classification layers on top of it
#x = Flatten()(last)
#x = Dense(256, activation='relu')(x)
#x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(last)


model = Model(base_model.input, output)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(learning_rate=1e-4,momentum=0.9), metrics=['accuracy'])
model.summary()


history1=model.fit(train_generator,
          validation_data=test_generator,
          epochs=100,
          verbose=1)


# Final evaluation of the model
scores = model.evaluate(test_generator, verbose=0)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])


# summarize history for accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
plt.savefig('C:/Users/DKU/Desktop/황반변성_CNN img/augmentSet_mobile1.png')
