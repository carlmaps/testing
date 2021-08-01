import numpy as np
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_topN(model, userID, config):
    
    logger.info("Retrieving Top 10 Movie Recommendation....")

    unique_ids = config.movieLensDF['itemID'].unique()
    
    # get the list of the ids that the userid 1001 has rated
    iids = config.movieLensDF.loc[config.movieLensDF['userID']==userID, 'itemID']
    
    # remove the rated movies for the recommendations
    movies_to_predict = np.setdiff1d(unique_ids,iids)
    
    my_recs = []
    for iid in movies_to_predict:
        my_recs.append((iid, model.predict(uid=userID,iid=iid).est))

    rawDF =  pd.DataFrame(my_recs, columns=['iid', 'predictions']).sort_values('predictions', ascending=False).head(10)['iid']
    return rawDF.tolist()

# def getMovieRecommendation(model, userID, config):

#     logger.info("Retrieving Top 10 Movie Recommendation....")

#     movies_watched_by_user = config.rating_df[config.rating_df.userId == userID]
#     movies_not_watched = config.movie_df[~config.movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
#     movies_not_watched = list(set(movies_not_watched).intersection(set(config.movie2movie_encoded.keys())))
#     movies_not_watched = [[config.movie2movie_encoded.get(x)] for x in movies_not_watched]
#     user_encoder = config.user2user_encoded.get(userID)
#     user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))

#     ratings = model.predict(user_movie_array).flatten()
#     top_ratings_indices = ratings.argsort()[-10:][::-1]
#     recommended_movie_ids = [config.movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

#     recommended_movies = config.movie_df[config.movie_df["movieId"].isin(recommended_movie_ids)]

#     recommendation = []
#     for row in recommended_movies.itertuples():
#         recommendation.append({"movieID" : row.movieId,"Title" : row.title,"Genre" : row.genres})
        
#     return recommendation

# def getRating(model, userID, movieID):
#      rating = model.predict(userID, movieID)
#      return {"rating" : rating.est}
    

