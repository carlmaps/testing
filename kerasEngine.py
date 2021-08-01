import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):

        logger.info("Starting up the RecommenderNet: ")

        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

    # The rate function to predict user's rating of unrated items
    def predictRate(self, userId, movieId):
        return {"rating": self.predict(np.array([[userId, movieId]])).astype(str).flatten()[0]}

    def getMovieRecommendation(self, userID, config):

        logger.info("Retrieving Top 10 Movie Recommendation....")

        movies_watched_by_user = config.rating_df[config.rating_df.userId == userID]
        movies_not_watched = config.movie_df[~config.movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
        movies_not_watched = list(set(movies_not_watched).intersection(set(config.movie2movie_encoded.keys())))
        movies_not_watched = [[config.movie2movie_encoded.get(x)] for x in movies_not_watched]
        user_encoder = config.user2user_encoded.get(userID)
        user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))

        ratings = self.predict(user_movie_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_movie_ids = [config.movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

        recommended_movies = config.movie_df[config.movie_df["movieId"].isin(recommended_movie_ids)]

        recommendation = []
        for row in recommended_movies.itertuples():
            recommendation.append({"movieID" : row.movieId,"Title" : row.title,"Genre" : row.genres})
            
        return recommendation
