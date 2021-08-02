from svdEngine import SVDEngine
from config import MovieRecommenderSetting
from kerasEngine import RecommenderNet
# from request import getRating, get_topN
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import os
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from flask_caching import Cache
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
cache = Cache(app,config={'CACHE_TYPE': 'simple'})
api = Api(app)

CSV_FILE = 'assets/movieLens.csv'
FILENAME = 'assets/SVDModel.sav'

global mvConfig
mvConfig = MovieRecommenderSetting()
global kerasModel
kerasModel = mvConfig.loadNNModel()
global svdModel
svdModel =  mvConfig.loadSVDModel(FILENAME)
global svdengine
svdengine = SVDEngine(mvConfig.ratingDF)
    

@app.route('/')
def home():
   return render_template('index.html')

class SVDRecommend(Resource):
    # def get(self, userID):
    #     return jsonify({'movies': get_topN(svdModel, userID, mvConfig)})
    #for SVD surprise
    def post(self):
        json_data = request.get_json(force=True)
        userid = json_data['userid']
        # return jsonify({'movies': svdengine.get_topN(svdModel, userid, mvConfig)})
        return jsonify(svdengine.recommend_movies(userid, mvConfig.movie_df, mvConfig.ratingDF))

class SVDRating(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        userid = json_data['userid']
        movieid = json_data['movieid']
        return jsonify(svdengine.getRating(svdModel, userid, movieid))
        # return jsonify(getRating(svdModel, userid, movieid))

class NeuralNetRecommend(Resource):
    # def get(self, userID):
    #     return jsonify(getMovieRecommendation(kerasModel, userID, mvConfig))
    def post(self):
        json_data = request.get_json(force=True)
        userid = json_data['userid']
        # return jsonify(getMovieRecommendation(kerasModel, userid, mvConfig))
        return jsonify(kerasModel.getMovieRecommendation(userid, mvConfig))

class NeuralNetRating(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        userid = json_data['userid']
        movieid = json_data['movieid']
        return jsonify(kerasModel.predictRate(userid, movieid))


# api.add_resource(Home, '/')
api.add_resource(SVDRecommend, '/svdrecommend')
api.add_resource(NeuralNetRecommend, '/nnrecommend')  
api.add_resource(SVDRating, '/svdrating')
api.add_resource(NeuralNetRating, '/nnrating')

if __name__ == "__main__":
    logger.info("Starting up the Movie Recommender API: ")
    # global mvConfig
    # mvConfig = MovieRecommenderSetting()
    # global kerasModel
    # kerasModel = mvConfig.loadNNModel()
    # global svdModel
    # # svdModel =  mvConfig.loadSVDModel()
    # global svdengine
    # svdengine = SVDEngine(mvConfig.ratingDF)
    app.run(debug=False)
