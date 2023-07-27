# Q3_graded
# Do not change the above line.
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

y_train = to_categorical(y_train, num_classes=10)  # convert the numerical label to one-hot encoding
y_test = to_categorical(y_test, num_classes=10)  # convert the numerical label to one-hot encoding

print(x_train[0].shape)

multi_layer = [3, 2, 1]  # compare layers with 3 different hidden value 3 layer , 2 layer and 1 layer
x_train = x_train / 255  # scaling data to fit model better
x_test = x_test / 255  # scaling data to fit model better


def train_model(layers, momentum=False, decay=False):
    model = tf.keras.Sequential()
    # first layer
    model = tf.keras.Sequential([
        Input(shape=x_train[0].shape),  # input layer 32 * 32 * 3 which is image height and width
        Flatten(),  # convert 3d array to 1
    ])
    for i in range(layers):
        if not decay:
            model.add(Dense(256, activation="relu"))  # add dense layer
        else:
            model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.l2(1e-4),
                            activity_regularizer=regularizers.l2(
                                1e-5)))  # add dense layer with weight decay(regularization)
    if not decay:
        model.add(Dense(10, activation="softmax"))  # add output softmax layer
    else:
        model.add(Dense(10, activation="softmax", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)))  # add output softmax layer with regularization

    moementum = momentum if momentum else 0
    # compile model
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=momentum),
                  # stochastic gradient descent optimizer with learning rate 0.001
                  loss='categorical_crossentropy',  # loss function
                  metrics=['accuracy']  # metrics to evaluate model
                  )

    result = model.fit(x_train, y_train, batch_size=128, epochs=60, verbose=1, validation_split=0.3)  # fit model
    result = result.history
    plt.figure()
    plt.plot(result["val_accuracy"])
    plt.title(f"accuracy with {layers} layers")
    plt.figure()
    plt.plot(result["val_loss"])
    plt.title(f"loss with {layers} layers")
    plt.show()
    print("----------------------------------------")


# uncomment to check different layer outcome
# for i in range(3, 0, -1):
#     train_model(i)

# uncomment to check result with momentum
# activating momentum
# train_model(3, 0.9)

# activating weight decay
train_model(3, 0.9, True)

