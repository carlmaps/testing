from kerasEngine import RecommenderNet
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EMBEDDING_SIZE = 50

class MovieRecommenderSetting:
    def __init__(self):

        logger.info("Starting up the MovieRecommenderSetting: ")

        # url1 = "https://raw.githubusercontent.com/carlmaps/ML-MovieRecommendation/master/assets/dataset/movies.csv"
        # url2 = "https://raw.githubusercontent.com/carlmaps/ML-MovieRecommendation/master/assets/dataset/m_rating.csv"
        # url3 = "https://raw.githubusercontent.com/carlmaps/ML-MovieRecommendation/master/assets/dataset/movieLens.csv"
        # url4 = "https://raw.githubusercontent.com/carlmaps/ML-MovieRecommendation/master/assets/dataset/ratings.csv"

        # self.movie_df = pd.read_csv(url1)
        # self.rating_df = pd.read_csv(url2)
        # self.movieLensDF = pd.read_csv(url3)
        # self.ratingDF = pd.read_csv(url4)

        self.movie_df = pd.read_csv("assets/dataset/movies.csv")
        self.rating_df = pd.read_csv("assets/dataset/m_rating.csv")
        self.movieLensDF = pd.read_csv("assets/dataset/movieLens.csv")
        self.ratingDF = pd.read_csv("assets/dataset/ratings.csv")

        user_ids = self.rating_df["userId"].unique().tolist()
    
        self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        self.userencoded2user = {i: x for i, x in enumerate(user_ids)}   
        self.movie_ids = self.rating_df["movieId"].unique().tolist()
        self.movie2movie_encoded = {x: i for i, x in enumerate(self.movie_ids)}
        self.movie_encoded2movie = {i: x for i, x in enumerate(self.movie_ids)}

        self.numOfUsers = len(self.user2user_encoded)
        self.numOfMovies = len(self.movie_encoded2movie)
        

    def __getUser2user_encoded(self):
        return self.user2user_encoded
    
    def __getMovie2movie_encoded(self):
        return self.movie2movie_encoded

    def __getMovie_encoded2movie(self):
        return self.movie_encoded2movie

    def loadSVDModel(self, filename):
        logger.info("Loading the Neural Network model")
        model = joblib.load(open(filename))
        # # _, model = dump.load('assets/SVDmodel2')
        return  model
    
    def loadNNModel(self):

        logger.info("Loading the Neural Network model")

        self.rating_df = self.rating_df.sample(frac=1, random_state=42)
        min_rating = min(self.rating_df["rating"])
        max_rating = max(self.rating_df["rating"])
        x = self.rating_df[["user", "movie"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = self.rating_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int(0.9 * self.rating_df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],)

        logger.info("Retraining the model on at least 1 data prior to loading the weight.")
        
        kerasModel = RecommenderNet(self.numOfUsers, self.numOfMovies, EMBEDDING_SIZE)
        kerasModel.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001))
        kerasModel.train_on_batch(x_train[:1], y_train[:1])
        kerasModel.load_weights('assets/model_weights')

        return kerasModel



    

        

    