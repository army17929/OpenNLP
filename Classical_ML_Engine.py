from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB 
import re
from Dataset_generation import load_preprocessed_nuclear_data
from Trainer import metrics_generator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time
import pandas as pd

    # Function for the pre-processing
def preprocess_text(text):
    # Conversion to the lowercase.
    text = text.lower()
    # Remove all the special characters.
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove all the stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemmize and Lemmatize
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

def vectorize_data(df,text_column:str):

    # Preprocess the data
    df[text_column] = df[text_column].apply(preprocess_text)

    # Define input and output
    X = df[text_column]

    # Vectorization
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
        
    return X_vectorized


class ClassicalML():
    def __init__(self,input_col:str,
                 label_col:str,
                 df,seed:int):
        self.X=vectorize_data(df=df,text_column=input_col) # Input data
        self.y=df[label_col] # Output data
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(
            self.X,self.y,test_size=0.2,random_state=seed)

    def run_RandomForest(self,n_estimators:int):
        # Random Forest Classifier model compile
        model = RandomForestClassifier(n_estimators=n_estimators)  # You can adjust the number of estimators as needed
        start=time.time()
        model.fit(self.X_train, self.y_train)
        end=time.time()
        print(f"RUNTIME : {end-start}")
        # Prediction
        y_pred = model.predict(self.X_test)
        metrics_generator(y_true=self.y_test,
                          y_pred=y_pred,
                        save_dir=f"/RandomForest_estimator{n_estimators}",
                        model_name='Random Forest')

    def run_DecisionTree(self):
        # Decision Tree classifier 
        model=DecisionTreeClassifier(criterion='entropy',
                                    splitter='best')
        start=time.time()
        model.fit(self.X_train,self.y_train)
        end=time.time()
        print(f"RUNTIME : {end-start}")

        # Prediction 
        y_pred=model.predict(self.X_test)
        metrics_generator(y_true=self.y_test,y_pred=y_pred,
                    save_dir=f"/DecisionTree",
                    model_name='DecisionTree')

    def run_MNB(self,alpha):
        model=MultinomialNB(alpha=alpha)
        start=time.time()
        model.fit(self.X_train,self.y_train)
        end=time.time()
        print(f"RUNTIME : {end-start}")

        y_pred=model.predict(self.X_test)
        metrics_generator(y_true=self.y_test,y_pred=y_pred,
                        save_dir=f"/MNB",
                            model_name='MNB')

    def run_GradBoost(self,n_estimators:int):
        model=GradientBoostingClassifier(n_estimators=n_estimators)
        start=time.time()
        model.fit(self.X_train,self.y_train)
        end=time.time()
        print(f"RUNTIME : {end-start}")

        y_pred=model.predict(self.X_test)
        metrics_generator(y_true=self.y_test,y_pred=y_pred,
                        save_dir=f"/GradBoost_estimator{n_estimators}",
                            model_name='GradBoost')

    def run_AdaBoost(self,n_estimators:int):
        model=AdaBoostClassifier(n_estimators=n_estimators)
        start=time.time()
        model.fit(self.X_train,self.y_train)
        end=time.time()
        print(f"RUNTIME : {end-start}")
        y_pred=model.predict(self.X_test)
        metrics_generator(y_true=self.y_test,y_pred=y_pred,
                        save_dir=f"/AdaBoost_estimator{n_estimators}",
                        model_name='AdaBoost')    
        
    def run_SVC(self):
        # Decision Tree classifier 
        model=SVC(kernel='linear')

        start=time.time()
        model.fit(self.X_train,self.y_train)
        end=time.time()
        print(f"RUNTIME : {end-start}")

        # Prediction 
        y_pred=model.predict(self.X_test)
        metrics_generator(y_true=self.y_test,y_pred=y_pred,
                    save_dir=f"/LinearSVC",
                    model_name='LinearSVC')

ML=ClassicalML(input_col='tweets',
               label_col='FinalScore',
               df=load_preprocessed_nuclear_data(),
               seed=42)

if __name__=="__main__":
    ML.run_AdaBoost(n_estimators=100)
    ML.run_GradBoost(n_estimators=100)
    ML.run_DecisionTree()
    ML.run_RandomForest(n_estimators=100)
    ML.run_SVC()
    ML.run_MNB(alpha=1.0)