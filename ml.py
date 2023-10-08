from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB 
import re
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
from data import load_preprocessed_nuclear_data
from trainer import metrics_generator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time

class ClassicalML():
    """
    Classical classification ML models.

    :param df: (DataFrame) data for model training
    :param input_col: (str) name of the column that contains input data
    :param output_col: (str) name of the column that contains output data
    :param seed: (int) random seed number for train and test split
    :param test_size: (float) portion of test data
    """
    def __init__(self,df,input_col:str,
                 output_col:str,
                 seed:int,test_size=0.2):
        self.X=self.vectorize_data(df=df,text_column=input_col) # Input data
        self.y=df[output_col] # Output data
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(
            self.X,self.y,test_size=test_size,random_state=seed)
        
    def preprocess_text(self,text):
        #This function pre-processes the raw text.
        #:params text: (str) string you want to process
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Stemmize and Lemmatize
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        return ' '.join(stemmed_tokens)

    def vectorize_data(self,df,text_column:str):
        #"""
        #This function vectorizes texts using TfidVectorizer.

        #:param df: (DataFrame) dataframe for model training 
        #:param text_column: (str) name of the column that contains text
        #"""
        # Preprocess the data
        df[text_column] = df[text_column].apply(self.preprocess_text)
        # Define input and output
        X = df[text_column]
        # Vectorization
        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        return X_vectorized

    def run_RandomForest(self,n_estimators:int):
        #"""
        #This function runs Random Forest model for sentiment classification.

        #:param n_estimators: (int) number of decision tree regressors
        #"""
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
        #"""
        #This function runs Decision Tree model for sentiment classification.
        #""" 
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

    def run_MNB(self,alpha:float):
        #"""
        #This function runs Multinomial Naive Bayes classifier model for sentiment classification.
        
        #:param alpha: (float) Additive smoothing parameter (set alpha=0 for no smoothing)
        #"""
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
        """
        This function runs Gradient Boosting classifier for sentiment classification.

        :param n_estimator: (int) The number of boosting stages to perform. Values must be in the range ``[1,inf)``
        """
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
        """
        This function runs AdaBoost classifier for sentiment classification. 
        An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset 
        and then fits additional copies of the classifier on the same dataset 
        but where the weights of incorrectly classified instances are adjusted 
        such that sunsequent classifiers focus more on difficult cases.

        :param n_estimators: (int) The maximum number of estimators at which boosting is terminated.
        """
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
        #"""
        #This function runs Linear Support Vector Classifier for sentiment classification.
        #"""
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
