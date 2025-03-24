'''
🚀 Excited to Share My Latest Project on Text Classification Using Machine Learning! 🚀

In the digital era, understanding textual data is more crucial than ever. I've been working on a text classification project that tackles sentiment analysis in restaurant reviews, aiming to decipher whether a review is positive or negative.

Project Overview:
The goal was to develop a robust model that could accurately classify textual sentiments, which can be pivotal for businesses in understanding customer satisfaction and improving services.

Technologies Used:
Python: For scripting and data manipulation.

NLTK: For natural language processing tasks like tokenization and stopwords removal.

Scikit-learn: Utilized for various ML models and preprocessing techniques.

XGBoost & LightGBM: Advanced machine learning frameworks for building efficient models.

CountVectorizer & TfidfVectorizer: For transforming text into a meaningful vector format.

Approach:
Data Preprocessing: Cleaned and prepared text data for modeling.

Feature Engineering: Used both Bag of Words and TF-IDF for feature extraction.

Model Training: Experimented with multiple classifiers including Logistic Regression, SVM, RandomForest, and more.

Optimization: Applied GridSearchCV for hyperparameter tuning to enhance model performance.

Evaluation: Assessed models based on accuracy and confusion matrix results.

Insights:
The project not only reinforced my skills in NLP and machine learning but also provided valuable insights into the effectiveness of different algorithms and techniques in handling real-world data.

I am thrilled to see how leveraging machine learning can significantly impact data-driven decision-making. A special thanks to everyone who supported and provided feedback throughout this journey.

🔗 [Link to the project or repository]

Feel free to reach out if you're interested in collaborating or learning more about this project!

#DataScience #MachineLearning #NaturalLanguageProcessing #AI #TechForGood
'''
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB

# Load dataset
ds = pd.read_csv(r"C:\Users\prasu\DS2\NLP\5th, 6th - NLP project\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the text
nltk.download('stopwords')
corpus = []
ps = PorterStemmer()
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', ds['Review'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(review))

# Define vectorizers for BoW and TF-IDF
vectorizers = {
    "BoW": CountVectorizer(max_features=1500),
    "TF-IDF": TfidfVectorizer(max_features=1500)
}

# Define models and parameters for GridSearchCV
models_params = {
    "Logistic Regression": (LogisticRegression(), {'C': [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "Random Forest": (RandomForestClassifier(), {'n_estimators': [50, 100, 150]}),
    "Decision Tree": (DecisionTreeClassifier(), {'max_depth': [10, 20, 30]}),
    "SVM": (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {'learning_rate': [0.01, 0.1, 0.2]}),
    "LGBM": (LGBMClassifier(), {'num_leaves': [20, 31, 40]}),
    "Naive Bayes": (MultinomialNB(), {'alpha': [0.5, 1.0, 1.5]})
}

# Loop through each vectorizer (BoW, TF-IDF) and each model
for vec_name, vectorizer in vectorizers.items():
    print(f"\n--- Vectorizer: {vec_name} ---\n")
    X = vectorizer.fit_transform(corpus).toarray()
    y = ds.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Loop through each model
    for model_name, (model, params) in models_params.items():
        print(f"\nModel: {model_name}")
        grid = GridSearchCV(model, params, scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        
        # Best parameters and accuracy
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best Training Score: {grid.best_score_}")

        # Test set predictions and metrics
        y_pred = grid.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_mat}")
        

