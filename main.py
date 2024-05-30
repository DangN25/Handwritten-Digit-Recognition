#===================================IMPORT===================================#
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 


#===================================TRAINNING/TESTING-DATA===================================#
mnist_data= tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test)= mnist_data.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test= tf.keras.utils.normalize(x_test, axis=1)


#===================================MODEL===================================#
model = tf.keras.models.Sequential()

#===================================LAYERS-SETUP===================================#
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#===================================COMPILE===================================#
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train,y_train, epochs=3)

model.save('handwritten.model.keras')


#===================================LOAD-MODEL===================================#
model= tf.keras.models.load_model("handwritten.model.keras")


#===================================PREDICT===================================#
im_num = 1
while os.path.isfile(f"digits/digit{im_num}.png"):
    try:
        img = cv2.imread(f"digits/digit{im_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction= model.predict(img)
        print(f"This digit is a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        im_num+=1