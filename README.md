# Airbnb-NYC-EDA-Price-Prediction-
Pricing a rental property on Airbnb is a challenging task for the owner as it determines the number of customers for the place. On the other hand, customers have to evaluate an offered price with minimal knowledge of an optimal value for the property.
Paper link: https://arxiv.org/pdf/1907.12665v1
dataset link: https://drive.google.com/drive/folders/1xk5RyR-UgF6M-ddhn11SXHEWJeB0fQo5

###########################################

Airbnb Price Prediction Using MachineLearning and Sentiment Analysis

Authors:

Kamran Ullah Afaq (kamranullahafaq@gmail.com)

Link to source paper for citation: https://arxiv.org/abs/1907.12665

###########################################

In order to run the code make sure you pre-instal all the dependecies such as TextBlob and sklearn

DOWNLOAD THE DATASET:
Create a directory called "Data", and download the datasets from this link into the directory: https://drive.google.com/drive/folders/1xk5RyR-UgF6M-ddhn11SXHEWJeB0fQo5?usp=sharing

INITIAL DATA PREPROCESSING:
Generate a fine with review sentiment: python sentiment_analysis.py
Clean the data: python data_cleanup.py
Normalize and split the data: data_preprocessing_reviews.py
GENERATE THE FEATURE SELECTION .NPY FILE:
For P-value feature selection: python feature_selection.py
For Lasso CV: python cv.py
TRAIN AND RUN THE MODELS: python run_models.py Note that by commenting/uncommenting certain lines of code you will be able to run different configurations of the models.
To run the models with Lasso CV feature selection comment out line 240 coeffs = np.load('../Data/selected_coefs_pvals.npy') and uncomment line 241 coeffs = np.load('../Data/selected_coefs.npy').
To run the models with p-value feature selection uncomment line 240 coeffs = np.load('../Data/selected_coefs_pvals.npy') and comment out line 241 coeffs = np.load('../Data/selected_coefs.npy').
To run the baseline uncomment the lines 277, 278
    print("--------------------Linear Regression--------------------")
    LinearModel(X_concat, y_concat, X_test, y_test)
and comment out everything below these lines. Also, comment out the lines 268, 269 and 270

   X_train = X_train[list(col_set)]
   X_val = X_val[list(col_set)]
   X_test = X_test[list(col_set)]
Warning: certain models take a while to train and run!
