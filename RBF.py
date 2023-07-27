import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
hidden_neurons_count = 3  # radial neuron count
data_counts = 200
X = np.arange(0, data_counts) / data_counts
Mio = (np.random.rand(data_counts) * 1.4 - 0.7) * 0
y = 1 / 3 + np.sin(3 * np.pi * X) + Mio
w = np.random.randn(hidden_neurons_count)
b = np.random.randn(1)
learning_rate = 0.01
centers = np.random.choice(X, size=hidden_neurons_count)
deviations = np.zeros(hidden_neurons_count)
epochs = 20000
precision = 0.00001


def gaussian(samples):
    distances = samples[:, np.newaxis] - centers[np.newaxis, :]
    distances = - (distances ** 2)
    distances = distances / (2 * deviations ** 2)
    distances = np.exp(distances)
    return distances


def final_product(distances):
    return distances.dot(w) + b


def fit():
    global w, b, centers, deviations
    copied_centers = centers.copy()
    converged = False

    while not converged:
        distances = np.abs(X[:, np.newaxis] - centers[np.newaxis, :])

        best_center_points = np.argmin(distances, axis=1)

        for i in range(hidden_neurons_count):
            each_center_points = X[best_center_points == i]
            if len(each_center_points) > 0:
                centers[i] = np.average(each_center_points)

        converged = np.linalg.norm(centers - copied_centers) < precision
        copied_centers = centers.copy()

    distances = np.abs(X[:, np.newaxis] - centers[np.newaxis, :])
    best_center_points = np.argmin(distances, axis=1)

    for i in range(hidden_neurons_count):
        each_center_points = X[best_center_points == i]
        if len(each_center_points) > 0:
            deviations[i] = np.std(each_center_points)

    # training
    for epoch in range(epochs):
        radial_neuron_result = gaussian(X)
        predicted_output = final_product(radial_neuron_result)
        error = predicted_output - y
        update_term_w = np.mean(learning_rate * radial_neuron_result * error[:, np.newaxis], axis=0)
        update_term_b = np.mean(learning_rate * error[:, np.newaxis], axis=0)
        w = w - update_term_w
        b = b - update_term_b


# Gaussian RBF


def predict(X):
    radial_output = gaussian(X)
    return final_product(radial_output)


fit()

predictions = predict(X)

plt.plot(X, predictions)
plt.plot(X, y)
plt.legend(["predict", "sinx"])
plt.xlabel('RBF')
plt.show()


