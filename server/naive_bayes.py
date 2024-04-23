import math

class NaiveBayes:
    
    def __init__(self):
        """ Initialize the model. """
        self.classes_ = None
        self.mean_ = None
        self.variance_ = None
        self.priors_ = None
        
    def fit(self, attributes, labels):
        """ Fit the model to the training data.

        Args:
            attributes (list of list of float): Training data attributes.
            labels (list of int): Training data labels.
        """
        # Find unique classes and their counts
        self.classes_ = sorted(set(labels))
        num_attributes = len(attributes[0])

        # Initialize statistics for each class
        self.mean_ = [[0 for _ in range(num_attributes)] for _ in self.classes_]
        self.variance_ = [[0 for _ in range(num_attributes)] for _ in self.classes_]
        self.priors_ = [0 for _ in self.classes_]

        # Calculate statistics for each class
        for class_index, class_value in enumerate(self.classes_):
            # Extract attribute for samples of the current class
            class_attributes = [attribute for attribute, label in zip(attributes, labels) if label == class_value]

            # Calculate mean and variance for each attribute
            for i in range(num_attributes):
                attribute_values = [attribute[i] for attribute in class_attributes]
                self.mean_[class_index][i] = sum(attribute_values) / len(attribute_values)
                self.variance_[class_index][i] = sum((x - self.mean_[class_index][i]) ** 2 for x in attribute_values) / (len(attribute_values) - 1)

            # Calculate prior probability for the class
            self.priors_[class_index] = len(class_attributes) / len(attributes)

    def _pdf(self, class_index, attribute_index, attribute_value):
        """ Calculate Gaussian probability density function for a given attribute.

        Args:
            class_index (int): Index of the current class.
            attribute_index (int): Index of the current attribute.
            attribute_value (float): Attributes value.

        Returns:
            float: Probability density of the attribute for the given class.
        """
        mean = self.mean_[class_index][attribute_index]
        stdev = math.sqrt(self.variance_[class_index][attribute_index])

        # Calculate the exponent term of the Gaussian PDF
        exponent = math.exp(-((attribute_value - mean) ** 2) / (2 * stdev ** 2))

        # Calculate the scaling factor (the denominator)
        scaling_factor = 1 / (math.sqrt(2 * math.pi) * stdev)

        # Return the Gaussian PDF value
        return scaling_factor * exponent

    def _predict(self, attribute):
        """This method calculates the likelihood of each class based on the given attributes and returns the class with the highest probability.

        Parameters:
            attribute (list of float): A list representing the attributes of a single data sample.

        Returns:
            int: The class label that has the highest probability for the given attributes.
        """
        # Initialize a list to store probabilities for each class
        class_probabilities = []

        # Iterate over each class to calculate its probability
        for class_index, _ in enumerate(self.classes_):
            # Start with the log of the prior probability of the class
            log_prior = math.log(self.priors_[class_index])

            # Sum the log of the probability density for each attribute
            log_likelihood = sum(
                math.log(self._pdf(class_index, attribute_index, attribute_value))
                for attribute_index, attribute_value in enumerate(attribute)
            )

            # Total log probability is the sum of log prior and log likelihood
            total_log_probability = log_prior + log_likelihood

            # Add the total log probability for this class to the list
            class_probabilities.append(total_log_probability)

        # Find the index of the class with the highest log probability
        most_probable_class_index = class_probabilities.index(max(class_probabilities))

        # Return the class label corresponding to the highest probability
        return self.classes_[most_probable_class_index]

    def predict(self, attributes):
        """ Predict the class labels for a list of attributes.

        Args:
            attributes (list of list of float): List of attribute vectors.

        Returns:
            list of int: Predicted class labels.
        """
        return [self._predict(attribute) for attribute in attributes]
