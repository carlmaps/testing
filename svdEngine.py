
from functools import cache
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from flask_caching import Cache

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cache = Cache()

class SVDEngine:
    def __init__(self, ratingsDF):

        logger.info("Initializing Engine....")

        ratings = ratingsDF.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
        self.allPredictions = self.getPredictionsAll(ratings)

    # @cache.cached(timeout = 10, key_prefix='getPredictionsAll')
    def getPredictionsAll(self,ratings):
        R = ratings.values #.as_matrix()
        user_ratings_mean = np.mean(R, axis = 1)
        Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(Ratings_demeaned, k = 50)
        sigma = np.diag(sigma)

        allPred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        return pd.DataFrame(allPred, columns = ratings.columns)

    def recommend_movies(self, userID, movies, original_ratings, num_recommendations=10):

        logger.info("Getting top 10 Recommendation....")
        
        # Get and sort the user's predictions
        user_row_number = userID - 1 # User ID starts at 1, not 0
        sorted_user_predictions = self.allPredictions.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
        
        # Get the user's data and merge in the movie information.
        user_data = original_ratings[original_ratings.userId == (userID)]
        user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').
                        sort_values(['rating'], ascending=False)
                    )

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
            merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                left_on = 'movieId',
                right_on = 'movieId').
            rename(columns = {user_row_number: 'Predictions'}).
            sort_values('Predictions', ascending = False).
                        iloc[:num_recommendations, :-1]
                        )
        recommendations = recommendations.head(10)
        recomList = []
        for index, row in recommendations.iterrows():
            recomList.append({"movieID" : row['movieId'],"Title" : row['title'],"Genre" : row['genres']})
            
        return recomList
    
    def getRating(self, model, userID, movieID):
        rating = model.predict(userID, movieID)
        return {"rating" : rating.est}