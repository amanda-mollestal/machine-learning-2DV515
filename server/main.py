from flask import Flask
from flask import jsonify
from flask_cors import CORS 
import csv
from naive_bayes import NaiveBayes

def load_dataset(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # Skip the header row
        dataset = [row for row in csv_reader]
    return dataset

def convert_class_labels_to_int(dataset, class_index):
    classes = sorted(set(row[class_index] for row in dataset))
    class_to_int = dict((c, i) for i, c in enumerate(classes))
    for row in dataset:
        row[class_index] = class_to_int[row[class_index]]
    return class_to_int  # To keep track of what integers map to what classes

# Load the datasets
iris_dataset = load_dataset('iris.csv')
banknote_dataset = load_dataset('banknote_authentication.csv')

# The last column is the class label
iris_class_to_int = convert_class_labels_to_int(iris_dataset, -1)
banknote_class_to_int = convert_class_labels_to_int(banknote_dataset, -1)

# Convert string values to float and class labels to int
iris_dataset = [[float(x) for x in row[:-1]] + [int(row[-1])] for row in iris_dataset]
banknote_dataset = [[float(x) for x in row[:-1]] + [int(row[-1])] for row in banknote_dataset]

# Separate the datasets into attributes (X) and labels (y)
x_iris, y_iris = [row[:-1] for row in iris_dataset], [row[-1] for row in iris_dataset]
x_banknote, y_banknote = [row[:-1] for row in banknote_dataset], [row[-1] for row in banknote_dataset]

def train_naive_bayes(x, y):
    model = NaiveBayes()
    model.fit(x, y)
    return model

# Train the models
iris_model = train_naive_bayes(x_iris, y_iris)
banknote_model = train_naive_bayes(x_banknote, y_banknote)

# Calculate accuracy
def accuracy_score(actual_labels, predicted_labels):
    correct_predictions = sum(actual == predicted for actual, predicted in zip(actual_labels, predicted_labels))
    return correct_predictions / len(actual_labels)

def confusion_matrix(actual, predicted, classes_to_int):
    # Initialize the confusion matrix with zeros
    num_classes = len(classes_to_int)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    class_names = [None] * num_classes  # List to hold class names ordered by index

    # Fill the confusion matrix and class names by comparing actual and predicted classes
    for a, p in zip(actual, predicted):
        matrix[a][p] += 1
        class_names[a] = list(classes_to_int.keys())[list(classes_to_int.values()).index(a)]

    # Pair each row of the matrix with the corresponding class name
    matrix_with_names = [{"class_name": class_names[index], "row": row} for index, row in enumerate(matrix)]

    return matrix_with_names

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the API!"

@app.route('/iris')
def iris():
    predictions = iris_model.predict(x_iris)
    accuracy = accuracy_score(y_iris, predictions)

    unique_classes_iris = len(iris_class_to_int)
    correct_predictions_iris = sum(1 for actual, predicted in zip(y_iris, predictions) if actual == predicted)
    conf_matrix = confusion_matrix(y_iris, predictions, iris_class_to_int)
    
    response_data = {
        "number_of_examples": len(x_iris),
        "number_of_attributes": len(x_iris[0]),
        "number_of_classes": unique_classes_iris,
        "accuracy": f"{accuracy:.2%}",
        "correctly_classified": f"{correct_predictions_iris}/{len(y_iris)}",
        "confusion_matrix": conf_matrix
    }

    return jsonify(response_data)

@app.route('/banknote')
def banknote():
    predictions = banknote_model.predict(x_banknote)
    accuracy = accuracy_score(y_banknote, predictions)

    unique_classes_banknote = len(banknote_class_to_int)
    correct_predictions_banknote = sum(actual == predicted for actual, predicted in zip(y_banknote, predictions))
    conf_metrix = confusion_matrix(y_banknote, predictions, banknote_class_to_int)

    response_data = {
        "number_of_examples": len(x_banknote),
        "number_of_attributes": len(x_banknote[0]),
        "number_of_classes": unique_classes_banknote,
        "accuracy": f"{accuracy:.2%}",
        "correctly_classified": f"{correct_predictions_banknote}/{len(y_banknote)}",
        "confusion_matrix": conf_metrix
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)