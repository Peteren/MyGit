#Import Image and numpy library
from PIL import Image
import numpy as np
import os
import cv2

#Code for making images into array
data=[]
labels=[]
cats=os.listdir("/home/nika90/Documents/animals/cats/")
for cat in cats:
    imag=cv2.imread("/home/nika90/Documents/animals/cats/"+cat)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((100, 100))
    data.append(np.array(resized_image))
    labels.append(0)
dogs=os.listdir("/home/nika90/Documents/animals/dogs")
for dog in dogs:
    imag=cv2.imread("/home/nika90/Documents/animals/dogs/"+dog)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((100, 100))
    data.append(np.array(resized_image))
    labels.append(1)
    
birds=os.listdir("/home/nika90/Documents/animals/birds")
for bird in birds:
    imag=cv2.imread("/home/nika90/Documents/animals/birds/"+bird)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((100, 100))
    data.append(np.array(resized_image))
    labels.append(2)
horses=os.listdir("/home/nika90/Documents/animals/horses")
for horse in horses:
    imag=cv2.imread("/home/nika90/Documents/animals/horses/"+horse)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((100, 100))
    data.append(np.array(resized_image))
    labels.append(3)

#Since the “data” and “labels” are normal array , convert them to numpy arrays
animals=np.array(data)
labels=np.array(labels)

#Now save these numpy arrays so that you dont need to do this image manipulation again
np.save("animals",animals)
np.save("labels",labels)

#Load the arrays
animals=np.load("animals.npy")
labels=np.load("labels.npy")

#Now shuffle the “animals” and “labels” set so that you get good mixture when you separate the dataset into train and test
s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

#Make a variable num_classes which is the total number of animal categories and a variable data_length which is size of dataset
num_classes=len(np.unique(labels))
data_length=len(animals)

#Divide data into test and train
(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

#Divide labels into test and train
(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

#Make labels into One Hot Encoding
import keras
from keras.utils import np_utils
#One hot encoding
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

#Making Keras model
#import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
#make model
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu",input_shape=(100, 100,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4,activation="softmax"))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
#Train the model
model.fit(x_train,y_train,batch_size=50
          ,epochs=100,verbose=1)

#Test the model
score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

model_json_str = model.to_json()
open('ani.json', 'w').write(model_json_str)
model.save_weights('ani.h5')

#Predicting on single images
def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((100, 100))
    return np.array(image)
def get_animal_name(label):
    if label==0:
        return "cat"
    if label==1:
        return "dog"
    if label==2:
        return "bird"
    if label==3:
        return "horse"
def predict_animal(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print(animal)
    print("The predicted Animal is a "+animal+" with accuracy =    "+str(acc))
	
predict_animal("/home/nika90/Documents/animals/singletest/testcat1.jpg")
predict_animal("/home/nika90/Documents/animals/singletest/testcat2.jpg")
predict_animal("/home/nika90/Documents/animals/singletest/testdog1.jpg")
predict_animal("/home/nika90/Documents/animals/singletest/testdog2.jpg")
predict_animal("/home/nika90/Documents/animals/singletest/testbird1.jpg")
predict_animal("/home/nika90/Documents/animals/singletest/testbird2.jpg")
predict_animal("/home/nika90/Documents/animals/singletest/testhorse1.jpg")
predict_animal("/home/nika90/Documents/animals/singletest/testhorse2.jpg")