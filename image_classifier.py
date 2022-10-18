#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sn; sn.set(font_scale=10)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm


# In[23]:


class_names = [ 'ECG', 'Prescription', 'Report', 'X-Ray']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

IMAGE_SIZE = (150, 150)


# In[24]:


# Loadind th data
def load_data():
    DIRECTORY = r"C:\computer vision\Basedata"
    CATEGORY = ["seg_train", "seg_test"]
    
    output = []
    
    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        print(path)
        images = []
        labels = []
        
        print("Loading {}".format(category))
        
        
        for folder in os.listdir(path):
            label = class_names_label[folder]
            
            #Iterate through each image in our folder
            for file in os.listdir(os.path.join(path, folder)):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(path, folder), file)
                
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32' )
    
        output.append((images, labels))
    
    return output


# In[25]:


(train_images, train_labels), (test_images, test_labels) = load_data()


# In[26]:


train_images, train_labels = shuffle(train_images, train_labels, random_state=25)


# In[27]:


def display_examples(class_names, images, labels):
    """
        Display 25 images from the images array with is corresponding labels
    """
    figsize = (200,200)
    fig = plt.figure(figsize = figsize)
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5,5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = cv2.resize(images[i], figsize)
        plt.imshow(image.astype(np.uint8))#, cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])         
    plt.show()
display_examples(class_names, train_images, train_labels)       


# In[28]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(10, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(101, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])


# In[29]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[30]:


history = model.fit(train_images, train_labels, batch_size=101, epochs=5, validation_split=0.5)


# In[31]:


def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(100,50))
    
    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    
    #Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    
    plt.legend()
    plt.show()


# In[32]:


plot_accuracy_loss(history)


# In[33]:


test_loss = model.evaluate(test_images, test_labels)


# In[34]:


predictions = model.predict(test_images)      # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
print(classification_report(test_labels, pred_labels))


# In[35]:


# using VGG16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=model.inputs, outputs=model.layers[-5].output)


# In[36]:


train_features = model.predict(train_images)
test_features = model.predict(test_images)


# In[37]:


from keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D, Flatten

model2 = VGG16(weights='imagenet', include_top=False)

input_shape = model2.layers[-4].get_input_shape_at(0) # get the input shape of desired layer
layer_input = Input(shape = (9, 9, 512)) # a new input tensor to be able to feed the desired layer
# https://stackoverflow.com/questions/5200025/keras-give-input-to-intermediate-layer-and-get-final-output

x = layer_input
for layer in model2.layers[-4::1]:
    x = layer(x)
    
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(100,activation='relu')(x)
x = Dense(6,activation='softmax')(x)

# create the model
new_model = Model(layer_input, x)


# In[38]:


new_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[39]:


history = new_model.fit(train_features, train_labels, batch_size=101, epochs=5, validation_split = 0.5)


# In[40]:


plot_accuracy_loss(history)


# In[41]:


from sklearn.metrics import accuracy_score

predictions = new_model.predict(test_features)
pred_labels = np.argmax(predictions, axis = 1)
print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))


# In[42]:


print(classification_report(test_labels, pred_labels))


# In[53]:





# In[56]:





# In[ ]:




