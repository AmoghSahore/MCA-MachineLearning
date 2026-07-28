# Lab 04 - Cell Explanations

## Cell 1 - Imports

**Explanation:** Imports the libraries needed for data handling, graphs, KNN, data splitting, scaling and evaluation. A fixed random state gives the same split each time.

**Important terms:**
- **Library:** A collection of ready-made functions.
- **Scikit-learn:** A Python library used for machine learning.
- **Random state:** A fixed seed that makes random results repeatable.

## Cell 2 - Loading the dataset

**Explanation:** Reads `brca.csv` into a DataFrame, renames the first column as `id`, and displays the first five rows.

**Important terms:**
- **CSV:** A text file that stores data in rows and columns.
- **DataFrame:** A table-like data structure in pandas.
- **ID:** A value used to identify each record.

## Cell 3 - Data checks

**Explanation:** Checks the dataset size, missing values, duplicate rows and the number of benign and malignant cases.

**Important terms:**
- **Shape:** The number of rows and columns in the dataset.
- **Missing value:** A data value that is not present.
- **Duplicate row:** A repeated record in the dataset.
- **Class distribution:** The number of records in each target class.

## Cell 4 - Features and target

**Explanation:** Converts the text labels into numbers. It then stores the input features in `X` and the target in `y`.

**Important terms:**
- **Encoding:** Converting category labels into numerical values.
- **Feature:** An input column used to make a prediction.
- **Target:** The output class that the model predicts.
- **X and y:** `X` contains features and `y` contains target values.

## Cell 5 - Split comparison

**Explanation:** Creates a reusable KNN pipeline and a function for metrics. It tests 80:20, 70:30 and 90:10 splits, chooses K using the square-root rule, trains each model and stores the results.

**Important terms:**
- **Pipeline:** Preprocessing and model steps applied together.
- **Standardization:** Changing features to a common scale.
- **Train-test split:** Dividing data into training and testing data.
- **Stratify:** Keeping a similar class ratio in both sets.
- **K:** The number of nearest neighbours used for prediction.

## Cell 6 - Heuristic K

**Explanation:** Creates the main 80:20 split and calculates an odd K value close to the square root of the training size.

**Important terms:**
- **Training set:** Data used to train the model.
- **Testing set:** Data used to check the trained model.
- **Heuristic:** A simple rule used to get a starting value.
- **Square-root rule:** K is estimated using the square root of the training size.

## Cell 7 - Nearby K values

**Explanation:** Trains KNN using values near the heuristic K. Accuracy, malignant recall and malignant F1 score are recorded for each K.

**Important terms:**
- **Accuracy:** The proportion of all predictions that are correct.
- **Recall:** The proportion of actual malignant cases correctly found.
- **F1 score:** The balance between precision and recall.
- **Hyperparameter:** A model setting chosen before training, such as K.

## Cell 8 - K and accuracy graph

**Explanation:** Plots accuracy against K to show how the number of neighbours affects the result.

**Important terms:**
- **Line plot:** A graph that connects values to show a trend.
- **Model selection:** Choosing the model setting that performs better.

## Cell 9 - Distance comparison

**Explanation:** Compares Euclidean and Manhattan distance using the same K and displays their scores.

**Important terms:**
- **Distance metric:** A method used to measure how close two points are.
- **Euclidean distance:** The straight-line distance between two points.
- **Manhattan distance:** The sum of the absolute feature differences.

## Cell 10 - Decision boundaries

**Explanation:** Uses two features to draw decision regions for K values 1, 5, 10 and 20. The background colour shows the predicted class in each region.

**Important terms:**
- **Decision boundary:** The border separating predicted classes.
- **Mesh grid:** A grid of points used to get predictions across a graph.
- **Overfitting:** When a model follows training data too closely.

## Cell 11 - Cross-validation

**Explanation:** Performs 5-fold stratified cross-validation for odd K values from 1 to 39 and selects the K with the highest average accuracy.

**Important terms:**
- **Cross-validation:** Testing a model several times on different data sections.
- **Fold:** One section of data used for validation.
- **Stratified K-Fold:** K-Fold that keeps the class ratio similar in every fold.
- **Standard deviation:** A measure of how much the fold scores vary.

## Cell 12 - Cross-validation graph

**Explanation:** Plots the mean cross-validation accuracy for every tested K value.

**Important terms:**
- **Mean CV accuracy:** The average accuracy across all folds.
- **Validation:** Checking performance on data not used for fitting in that fold.

## Cell 13 - Final model

**Explanation:** Trains the final KNN model using the best K. It predicts classes and malignant probabilities, then prints the main classification metrics.

**Important terms:**
- **Precision:** The proportion of predicted malignant cases that are actually malignant.
- **Probability score:** The model's confidence for a class.
- **ROC-AUC:** A score showing how well the model separates both classes.

## Cell 14 - Confusion matrix

**Explanation:** Creates a table showing correct and incorrect predictions for malignant and benign cases.

**Important terms:**
- **Confusion matrix:** A table comparing actual and predicted classes.
- **True positive:** A malignant case correctly predicted as malignant.
- **False negative:** A malignant case incorrectly predicted as benign.

## Cell 15 - Classification report

**Explanation:** Prints precision, recall, F1 score and support separately for both classes.

**Important terms:**
- **Classification report:** A summary of metrics for each class.
- **Support:** The number of actual samples in a class.
- **Macro average:** The simple average across classes.
- **Weighted average:** The average adjusted according to class size.

## Cell 16 - ROC curve

**Explanation:** Calculates and plots the ROC curve for the malignant class. The AUC value is shown in the graph legend.

**Important terms:**
- **ROC curve:** True positive rate plotted against false positive rate.
- **True positive rate:** Another name for recall or sensitivity.
- **False positive rate:** The proportion of benign cases incorrectly marked malignant.
- **Threshold:** The probability limit used to decide a class.
- **AUC:** Area under the ROC curve; a higher value means better separation.
