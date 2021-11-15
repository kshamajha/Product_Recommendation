#!/usr/bin/env python
# coding: utf-8

# Final Sentiment Based Product Recommendation Model with the code to deploy the end-to-end project using Flask and
# Heroku

# Suppress Warnings
import warnings
import numpy as np

warnings.filterwarnings('ignore')

import pickle as pkl

# import all pickle files
# best_model.pkl - sentiment analysis best model pickle file
# tfidf.pkl - tfidf vectorizer
# transform.pkl - the pickle file after text cleaning
# user_recommendation.pkl - user based recommendation model

model = pkl.load(open('models/best_model.pkl', 'rb'))
tfidf = pkl.load(open('models/tfidf.pkl', 'rb'))
transform = pkl.load(open('dataset/transform.pkl', 'rb'))
user_recom = pkl.load(open('models/user_recommendation.pkl', 'rb'))


# Function to get the top 20 products based on user recommendation
def recommendation(user_input):
    try:
        result = True
        recom_data = user_recom.loc[user_input].sort_values(ascending=False)[0:20].index
    except:
        result = False
        recom_data = "User with username \"" + user_input + "\" not found in dataset, try another username"
    return result, recom_data


# Function to get the top 5 products from the recommended 20 products
def sentiment(recom_prod):
    df = transform[transform.name.isin(recom_prod)]
    features = tfidf.transform(df['text'])
    pred_data = model.predict(features)
    predictions = [round(value) for value in pred_data]
    df['predicted'] = predictions
    grouped_df = df.groupby(['name'])
    product_class = grouped_df['predicted'].agg(mean_class=np.mean)
    df = product_class.sort_values(by=['mean_class'], ascending=False)[:5]
    df['Product Name'] = df.index
    output_data = df[['Product Name']][:5].reset_index(drop=True)
    return output_data
