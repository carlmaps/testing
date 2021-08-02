from svdEngine import SVDEngine
from config import MovieRecommenderSetting
from kerasEngine import RecommenderNet
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from flask_caching import Cache
from contentEngine import ContentBased
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
# global cbEngine
# cbEngine = ContentBased(mvConfig.movie_df, mvConfig.ratingDF)
    

@app.route('/')
def home():
   return render_template('index.html')

class Content(Resource):
    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self, title):
        result = cbEngine.get_recommendations_based_on_genres(title)
        # result = list(result)
        # dico = {}
        # for i in range(len(result)):
	    #     dico["title" + str(i+1)] = result[i]
        return jsonify(result)

class ContentRecommender(Resource):
  
    def get(self, num):
        result = cbEngine.get_recommendation_content_model(num)
        result = list(result)
        dico = {}
        for i in range(len(result)):
	        dico["title" + str(i+1)] = result[i]
        return jsonify(dico)

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
        # return jsonify(svdengine.getRating(userid, movieid, mvConfig.movie_df, mvConfig.ratingDF))
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
api.add_resource(Content, '/content/<title>')
api.add_resource(ContentRecommender, '/recommendById/<int:num>')
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
