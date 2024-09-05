
# 1. Import necessary libraries
import pandas as pd
import math
from sklearn.model_selection import train_test_split

# 2. Load the dataset
df = pd.read_csv('ecoli.data', delim_whitespace=True, header=None)

# 3. Preprocess the data
# Filter only the rows with 'cp' and 'im' labels
filtered_df = df[df[8].isin(['cp', 'im'])]

# Split data and labels
X = filtered_df.iloc[:, 1:8].values  # Using all the seven features columns
y = (filtered_df[8] == 'cp').astype(int).values  # Convert labels 'cp' and 'im' to 1 and 0

# Split dataset into training and testing sets (80-20% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #random_state=42)

# 4. Define the neural network
weights = [-0.1, 0.20, -0.23, -0.1, 0.20, -0.23, -0.1, 0.20, -0.23]

def sigmoid(z):
    if z < -100:
        return 0
    if z > 100:
        return 1
    return 1.0 / (1.0 + math.exp(-z))

def firstLayer(row, weights):
    activation_1 = weights[0] * 1
    for i in range(3):
        activation_1 += weights[i + 1] * row[i]
    activation_2 = weights[3] * 1
    for i in range(3, 6):
        activation_2 += weights[i + 1] * row[i]
    return sigmoid(activation_1), sigmoid(activation_2)

def secondLayer(row, weights):
    activation_3 = weights[6]
    activation_3 += weights[7] * row[0]
    activation_3 += weights[8] * row[1]
    return sigmoid(activation_3)

def predict(row, weights):
    input_layer = row
    first_layer = firstLayer(input_layer, weights)
    second_layer = secondLayer(first_layer, weights)
    return second_layer, first_layer

# 5. Train the neural network
def train_weights(train, labels, learning_rate, epochs):
    for epoch in range(epochs):
        for row, label in zip(train, labels):
            prediction, first_layer = predict(row, weights)
            error = label - prediction
            # First layer
            for i in range(4):
                weights[i] += learning_rate * error * row[i]
            # Second layer
            for i in range(2):
                weights[i + 6] += learning_rate * error * first_layer[i]
    return weights

learning_rate = 0.0001
epochs = 1000
trained_weights = train_weights(X_train, y_train, learning_rate, epochs)

# 6. Evaluate the neural network on the testing dataset
correct_predictions = 0
for row, label in zip(X_test, y_test):
    prediction, _ = predict(row, trained_weights)
    if round(prediction) == label:
        correct_predictions += 1

accuracy = correct_predictions / len(y_test) * 100
print(f"Accuracy on the testing dataset: {accuracy:.2f}%")

