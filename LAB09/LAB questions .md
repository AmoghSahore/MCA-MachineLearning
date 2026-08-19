## **Lab 9**

### **Aim**

To implement Support Vector Machine (SVM) for classification and Principal Component Analysis (PCA) for dimensionality reduction, and analyse their effectiveness using real world datasets.

### **Objectives**

1. To implement and evaluate the performance of the Support Vector Machine (SVM) classifier using different kernel functions.  
2. To compare the performance of SVM models using standard classification metrics.  
3. To apply Principal Component Analysis (PCA) for reducing the dimensionality of a high dimensional dataset.  
4. To analyse the variance retained by the principal components and visualize the transformed feature space.  
5. To understand the role of supervised learning (SVM) and unsupervised learning (PCA) in machine learning applications.

### **Part A: Support Vector Machine (SVM)**

**Aim:**  
 To implement the Support Vector Machine (SVM) classifier for a binary classification problem and evaluate its performance.

**Dataset:**  
 UCI Breast Cancer Wisconsin (Diagnostic) Dataset  
 [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Tasks:**

1. Load the dataset and perform the necessary preprocessing.  
2. Split the dataset into training and testing sets (80:20).  
3. Train an SVM classifier using a **Linear Kernel** and explore hyper parameter tuning.   
4. Evaluate the model using:  
   * Accuracy  
   * Precision  
   * Recall  
   * F1 Score  
   * Confusion Matrix  
5. Summarize your observations.

---

## **Part B: Principal Component Analysis (PCA)**

**Aim:**  
 To implement Principal Component Analysis (PCA) for dimensionality reduction and analyse the variance retained by the principal components.

**Dataset:**  
 UCI Wine Dataset  
 [https://archive.ics.uci.edu/dataset/109/wine](https://archive.ics.uci.edu/dataset/109/wine)

**Tasks:**

1. Load the dataset and perform feature standardization.  
2. Apply PCA to reduce the dataset from 13 features to 2 principal components.  
3. Display the explained variance ratio of each principal component.  
4. Calculate the cumulative explained variance and determine the minimum number of principal components required to retain at least **95%** of the total variance.  
5. Visualize the transformed dataset using a two dimensional scatter plot.  
6. Compare the original and transformed datasets in terms of:  
   * Number of features  
   * Information retained  
   * Computational efficiency  
7. Interpret the significance of the first two principal components.  
8. Discuss the advantages, limitations, and applications of PCA in machine learning.

**Extra credit  Task :**

Apply **Linear Discriminant Analysis (LDA)** to the same dataset and reduce it to two linear discriminants. Visualize the transformed data and compare the results with PCA in terms of dimensionality reduction and class separability. Summarize your observations.

