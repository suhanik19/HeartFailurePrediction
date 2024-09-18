# Heart Failure Prediction
Welcome to the Heart Disease Prediction Tool. This Python-based model is designed to predict heart disease by analyzing key health features. It trains multiple classification models to improve diagnostic accuracy, providing valuable insights for medical research and healthcare analytics.

## Project Objective
Heart disease remains one of the leading causes of mortality worldwide, making early detection and accurate diagnosis crucial for improving patient outcomes. With advancements in machine learning and data analytics, predictive models can significantly assist healthcare professionals by providing data-driven insights that aid in early diagnosis and personalized treatments. This project aims to address that need by leveraging a comprehensive heart disease dataset and applying classification models to predict the likelihood of heart disease.

### Dataset and Features
The dataset used in this project combines five existing heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog), resulting in 918 unique observations after removing any duplicate values. Each observation contains 11 key attributes that capture vital health information associated with heart disease:

1. **Age**: The patient’s age [years].

2. **Sex**: The gender of the patient (0 = female, 1 = male).

3. **Chest Pain Type**: Categorized into four chest pain types (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic).

4. **Resting Blood Pressure**: The patient’s resting blood pressure [mm Hg].

5. **Serum Cholesterol**: The cholesterol level [mg/dL]. 

6. **Fasting Blood Sugar**: The patient's blood sugar after fasting (1: if FastingBS > 120 mg/dl, 0: otherwise).

7. **Resting ECG**: Provides information about the patient’s heart at rest, including normal results, ST-T wave abnormalities, or showing left ventricular hypertrophy (LVH).

8. **Maximum HR**: The maximum heart rate reached during a stress test (Numeric value between 60 and 202).

9. **Exercise-Induced Angina**: Indicates whether the patient experienced angina (chest pain) during physical exertion (1 = yes, 0 = no).

10. **Old Peak**: ST depression induced by exercise relative to rest (Numeric value measured in depression).

11. **ST_Slope**: the slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping).

12. **Target**: The target variable (0 = no heart disease, 1 = presence of heart disease) indicates whether the patient has heart disease.

Every dataset used can be found under the Index of heart disease datasets from UCI Machine Learning Repository on the following link: [https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Methods Used
* **Data Preprocessing**: Handling missing values and duplicates, mapping categorical features to numerical values, scaling data with StandardScaler, and addressing class imbalance using Random OverSampling.
* **Exploratory Data Analysis (EDA)**: Visualizing feature distributions with histograms and analyzing correlations with a heatmap.
* **Classification Models**: Used Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (SVM), Gradient Boosting Machines (GBM)

### Technologies
* **Python**
* **Pandas**
* **Seaborn & Matplotlib**: for data visualization
* **NumPy**
* **Scikit-learn**: for machine learning models
* **Imbalanced-learn**

## Project Workflow Overview
1. **Data Preprocessing**: The dataset is clean, reducing the need for extensive preprocessing. Categorical features are mapped to numerical values, data scaling is performed using StandardScaler, and class imbalance is addressed with Random OverSampling to ensure balanced model training.
2. **Exploratory Data Analysis**: Visualizations are used to analyze feature distributions and relationships. Histograms are generated to compare the distributions of features across target classes, and a correlation heatmap is presented to highlight relationships between features, helping to identify any multicollinearity or key patterns.
3. **Model Training**: Six different machine learning models are trained on the processed dataset.
4. **Model Evaluation and Performance**: Each model is evaluated based on accuracy and classification metrics, with Random Forests and Gradient Boosting achieving the highest accuracy.

## Usage
1. **Access the Colab Notebook**:
Click [here](https://colab.research.google.com/drive/11wUp60sJ0i_OsW7F26zhi5gjmwDf4AUF?authuser=1#scrollTo=UmJT0Q6KgU-B) to open the project notebook in Google Colab.
2. **Run the Code**:
Once the notebook is open, you can run the code cells sequentially by clicking the "Run" button at the top of each cell or by selecting Runtime > Run All from the menu.
3. **Environment Setup**:
The Colab notebook is pre-configured with the necessary libraries and dependencies. When prompted, Colab will install any additional libraries automatically.
4. **Uploading the Dataset**:
If the dataset isn't automatically loaded in the notebook, upload the data files manually. Colab provides an easy way to do this by using the Upload button within the notebook. The dataset is linked above.
5. **Running the Models**:
Follow the steps outlined in the notebook to preprocess the data, train models, and evaluate their performance. Each section is clearly marked for easy navigation.

## Model Performance
This project applied six different machine learning classification models to predict the likelihood of heart disease based on 11 key health features. Each model was trained on the processed dataset and evaluated using the test data.

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 0.864130 |
| K-Nearest Neighbors | 0.864130 |
| Decision Trees | 0.836957 |
| Random Forests | 0.885870 |
| Support Vector Machine | 0.875000 |
| Gradient Boosting Machines | 0.891304 |

Both **Random Forests** and **Gradient Boosting Machines** delivered the highest accuracy scores, making them the most reliable models in this project for predicting heart disease. The models were evaluated using multiple metrics, including precision, recall, and F1-score. For simplicity, the table above shows the overall accuracy of each model. Detailed evaluation metrics are available in the code.

## Future Work
While the current models demonstrate promising results, there are several areas for further improvement:
* **Advanced Feature Engineering**:
   * Conduct recursive feature elimination to remove the least important features based on model performance.
   * Conduct feature importance analysis to determine which features contribute most to the model’s predictions and refine the feature set.
* **Cross Validation**:
   * Split the data into k subsets and train the model k times, each time using a different subset as validation data and the remaining data as training data. This will ensure the models generalize well across different subsets of the data.
* **Model Selection**:
   * Use stacking to combine multiple models to improve overall performance. The predictions of base models are used as input to a final model.
   * Use voting classifier to aggregate predictions from multiple models to make a final decision based on majority voting or averaging.

## Acknowledgements
Creators:

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.

2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D. 

3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.

4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

Donor: David W. Aha (aha@ics.uci.edu) (714) 856-8779

## Contact
For any issues or further inquiries, please contact Suhani Khandelwal at suhanikhnadelwal05@gmail.com. If you encounter any bugs or have suggestions, feel free to open an issue on the GitHub repository.
