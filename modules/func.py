import secrets
import string
import random
import os
import re
import pandas as pd
import africastalking
import streamlit as st 
import google.generativeai as genai


from collections import defaultdict, Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel



from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

africastalking.initialize(
    username='EMID',
    api_key = os.getenv("AT_API_KEY")
)

sms = africastalking.SMS
airtime = africastalking.Airtime
voice = africastalking.Voice




def send_sms(phone_number, otp_sms):
    # amount = "10"
    # currency_code = "KES"

    recipients = [f"+254{str(phone_number)}"]

    # airtime_rec = "+254" + str(phone_number)

    print(recipients)
    print(phone_number)

    # Set your message
    message = f"{otp_sms}";

    # Set your shortCode or senderId
    sender = 20880

    try:
        # responses = airtime.send(phone_number=airtime_rec, amount=amount, currency_code=currency_code)
        response = sms.send(message, recipients, sender)

        print(response)

        # print(responses)

    except Exception as e:
        print(f'Houston, we have a problem: {e}')

    st.toast(f"OTP Sent Successfully")



def make_call(phone_number):    
  
  # Set your Africa's Talking phone number in international format
    callFrom = "+254730731123"
  
  # Set the numbers you want to call to in a comma-separated list
    callTo   = [f"+254{str(phone_number)}"]
    
    try:
  # Make the call
        result = voice.call(callFrom, callTo)
        print (result)
    except Exception as e:
        print ("Encountered an error while making the call:%s" %str(e))



def generate_otp(length=6):
    characters = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

# print("Generated OTP:", generate_otp())

def load_dataframe():
    df = pd.read_csv(r"../src/food recipe/Food_Recipe.csv")
    return df


def clean_data(x):
        return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['name']+ ' ' +  x['cuisine'] + ' ' + x['course'] + ' ' + x['ingredients_name'] + ' ' + x['diet'] #+ ' ' + x['prep_time (in mins)'] + ' ' + x['cook_time (in mins)']

def recommend_food_general(name, cosine_sim):
    global result
    name=name.replace(' ','').lower()
    idx = food_indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:300]
    food_indices = [i[0] for i in sim_scores]
    result =  df[['name', 'diet', 'cuisine', 'course', 'ingredients_name', 'prep_time (in mins)', 'cook_time (in mins)', 'description']].iloc[food_indices]
    result.reset_index(drop=True, inplace=True)
    return result

df = load_dataframe()
df = df.fillna('')
# df['prep_time (in mins)'] = df['prep_time (in mins)'].apply(lambda x: str.lower(x))
# df['cook_time (in mins)'] = df['cook_time (in mins)'].apply(str.lower())


new_features = ['name', 'cuisine', 'course', 'ingredients_name', 'diet'] #, 'prep_time (in mins)', 'cook_time (in mins)']
food_data = df[new_features]

for i in new_features:
    food_data[i] = df[i].apply(clean_data)

food_data['soup'] = food_data.apply(create_soup, axis=1)

cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(food_data['soup'])

# global cosine_sim

cosine_sim = cosine_similarity(count_matrix, count_matrix)

food_data=food_data.reset_index()

indices = pd.Series(food_data.index, index=food_data['name'])
# st.write('Heres the indices series')
# st.dataframe(indices





