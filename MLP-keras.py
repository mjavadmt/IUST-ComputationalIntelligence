from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
np.random.seed(0)
data_counts = 800
X = np.arange(0, data_counts) / data_counts
Mio = (np.random.rand(data_counts) * 1.4 - 0.7) * 0
y = 1 / 3 + np.sin(3 * np.pi * X) + Mio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# Set the input shape
input_shape = (1,)
print(f'Feature shape: {input_shape}')


def actual_output(samples):
    return 1 / 3 + np.sin(3 * np.pi * samples)


# Create the model
model = Sequential()
model.add(Dense(16, input_shape=input_shape, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Configure the model and start training
model.compile(optimizer=SGD(learning_rate=0.01),
              # stochastic gradient descent optimizer with learning rate 0.001
              loss='mean_absolute_error',  # loss function
              metrics=['accuracy']  # metrics to evaluate model
              )
model.fit(X_train, y_train, epochs=300, batch_size=1, verbose=1, validation_data=(X_test, y_test))
sorted_X_train = np.sort(X_train)
sorted_X_valid = np.sort(X_test)
predicted_train = model.predict(sorted_X_train)
predicted_validation = model.predict(sorted_X_valid)
actual_train = actual_output(sorted_X_train)
actual_validation = actual_output(sorted_X_valid)

plt.plot(sorted_X_train, predicted_train, label="predicted")
plt.plot(sorted_X_train, actual_train, label="actual")
plt.legend(loc='best')
plt.title("train X comparison")

plt.figure()
plt.plot(sorted_X_valid, predicted_validation, label="predicted")
plt.plot(sorted_X_valid, actual_validation, label="actual")
plt.legend(loc='best')
plt.title("validation X comparison")

predicted = model.predict(X)
actual = y
plt.figure()
plt.plot(X, predicted, label="predicted")
plt.plot(X, actual, label="actual")
plt.legend(loc='best')
plt.title("total X comparison")
plt.show()
