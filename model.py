#!/usr/bin/env python
# coding: utf-8

# Final Sentiment and Recommendation Model with the code to deploy the end-to-end project using Flask and Heroku

# Suppress Warnings
import warnings

warnings.filterwarnings('ignore')

import pickle as pkl

# ## import all pickle files
# #### xbg.pkl - sentiment analysis XGBoost model pickle file
# #### tfidf.pkl - tfidf vectorizer
# #### transform.pkl - this pickle file after text cleaning
# #### user_recommendation.pkl - user based recommendation model

xgb = pkl.load(open('models/Xgboost.pkl', 'rb'))
tfidf = pkl.load(open('models/tfidf.pkl', 'rb'))
transform = pkl.load(open('dataset/transform.pkl', 'rb'))
user_recom = pkl.load(open('models/user_recommendation.pkl', 'rb'))


def sentiment(recom_prod):
    df = transform[transform.name.isin(recom_prod)]
    features = tfidf.transform(df['text'])
    pred_data = xgb.predict(features)
    predictions = [round(value) for value in pred_data]
    df['predicted'] = predictions
    output_data = df[df['predicted'] == 1][['name', 'brand', 'categories']].drop_duplicates()[:5].reset_index(drop=True)

    return output_data


def recommendation(user_input):
    try:
        result = True
        recom_data = user_recom.loc[user_input].sort_values(ascending=False)[0:20].index
    except:
        result = False
        recom_data = "User with username \"" + user_input + "\" not found, try another username"
    return result, recom_data