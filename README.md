# Classifying X-rays With Convolutional Neural Network

**Project Summary:**

In this project, the goal was to develop a robust image classification model using a dataset of X-ray images related to Covid-19, Normal, and Pneumonia cases. The project utilized deep learning techniques implemented in TensorFlow and Keras to train a neural network capable of accurately classifying X-ray scans into these three categories. Data was sourced from [Kaggel](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

**Project Overview:**  
*Exploration and Preprocessing:*

- The initial steps involved exploring the dataset, understanding its structure, and visualizing random examples from each class to gain insights into the data. The dataset was loaded and split into training and validation sets. Data augmentation techniques, such as random flips, rotations, and zooms, were applied to the training set to enhance model generalization. The data was also cached and prefetched for optimal performance during training.

*Model Building and Evaluation:*

- The neural network architecture consisted of convolutional layers, max-pooling layers, and dense layers for classification. The initial model achieved promising results, but further experimentation was conducted to improve performance. Additional layers were added, and hyperparameter tuning, including learning rate optimization, was performed to enhance the model's accuracy.

- The final model achieved a significant improvement in categorical accuracy, reaching 92.6% on the training set and 86.4% on the validation set. Hyperparameter tuning further boosted the accuracy on the validation set by approximately 4% on both sets.

*Model Evaluation:*

- The evaluation metrics, including loss, categorical accuracy, and AUC, were tracked over epochs. The model's performance was visualized through plots depicting accuracy and AUC trends. The results showed a clear improvement in performance, indicating the effectiveness of the model.

*Classification Report:*

- A detailed classification report was generated to assess the model's performance on individual classes (Covid, Normal, Pneumonia). Precision, recall, and F1-score were analyzed, providing insights into the model's ability to correctly identify and classify instances for each class. The model exhibited high precision and recall across all classes, demonstrating its reliability in differentiating between X-ray images related to various health conditions.

*Confusion Matrix:*

- A confusion matrix was created to delve deeper into the model's predictions. The matrix provided a breakdown of correct and incorrect predictions for each class, highlighting areas of potential improvement. Notable findings included instances of misclassification between Normal and Pneumonia cases.

**Key Insights:**

- The model showcased balanced performance, minimizing both false positives and false negatives.
- Specific areas for improvement were identified through the confusion matrix, guiding future refinements to enhance accuracy further.
- Overall, the model demonstrated its potential as a valuable tool in aiding healthcare professionals in the diagnosis of lung-related illnesses based on X-ray scans.

This project exemplifies the iterative process of model development, experimentation, and evaluation, emphasizing the importance of continuous refinement to achieve optimal performance.

----