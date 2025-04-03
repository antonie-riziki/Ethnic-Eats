import streamlit as st
import pandas as pd
import africastalking
import os
import sys
import requests
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
import csv
import time
import google.generativeai as genai



from bs4 import BeautifulSoup


# from collections import defaultdict, Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


sb.set()
sb.set_style('darkgrid')
sb.set_palette('viridis')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

warnings.filterwarnings('ignore')



sys.path.insert(1, './modules')
print(sys.path.insert(1, '../modules/'))

from func import generate_otp, send_sms, make_call#, load_dataframe, clean_data, create_soup, recommend_food_general

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

africastalking.initialize(
    username='EMID',
    api_key = os.getenv("AT_API_KEY")
)

sms = africastalking.SMS
airtime = africastalking.Airtime



st.image('https://img.sndimg.com/food/image/upload/f_auto,c_thumb,q_55,w_1280,ar_16:9/v1/img/recipes/30/08/1/uaSpHBwtSIuQpcHZKQeu_0S9A0050.jpg', width=1500)





############################################################################






# instances of the application

header = st.container()
data_loading = st.container()
feature_eng = st.container()
side_col = st.container()





# Header section

with header:
    pass
    # st.title('Netflix Movies Recommendation System')
    # st.image("Netflix-Continue-Watching.gif")
    # st.write('While recommendation systems have gained the popularity in the advanced of internet usage, there has been need for developing systems that will however meet user preferences')
    # st.write('On the other hand, we specified the choice of Content Based System rather than Colaborative Filtering')
    # st.markdown('Lets roll our sleeves and get it done')



with data_loading:
    # st.subheader('Data Loading and Exploration')
    df = pd.read_csv(r"./src/food recipe/Food_Recipe.csv")
    # df = load_dataframe()
    
    
    # st.write(df.head())

    with open("./src/food recipe/Food_Recipe.csv", "rb") as file:
        st.download_button(label = 'download csv file', data = file, file_name = "food_recipe.csv")
    
    # st.write('The dataframe has ' + str(df.shape[0]) + ' features and ' + str(df.shape[1]) + ' records')

    # st.text('Statistical Representation of the Dataset')
    # col1, col2 = st.columns(2)

    
    # with col1:
    
    #     st.text('Dataset Decsription')
    #     st.write(df.describe())

    # with col2:
    #     st.text('Inpect missing values (NaNs')
    #     st.write(df.isnull().sum())

    # st.text('Graphical Representation of Nan Values')
    # st.bar_chart(df.isnull().sum())

    # st.markdown('Since the dataset has NaN values, we will not clean since thats not the objective of this system, but for future accuracy of the model, we will have to.')



with st.sidebar:
    select_food = st.selectbox('select a food', (sorted(df['name'].unique())))

with feature_eng:
    # st.header('Feature Engineering')
    # st.text('Content-Based Recommendation System')
    # st.image("African-movies-on-Netflix.png")
    # st.write('We will use the TF-IDF vectorizer to evaluate the "overview" series and convert it to a Document Term Matrix for evaluation')


    def clean_data(x):
        return str.lower(x.replace(" ", ""))

    def create_soup(x):
        return x['name']+ ' ' +  x['cuisine'] + ' ' + x['course'] + ' ' + x['ingredients_name'] + ' ' + x['diet'] #+ ' ' + x['prep_time (in mins)'] + ' ' + x['cook_time (in mins)']

    def recommend_food_general(name, cosine_sim):
        global result
        name=name.replace(' ','').lower()
        idx = indices[name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:300]
        food_indices = [i[0] for i in sim_scores]
        result =  df[['name', 'diet', 'cuisine', 'course', 'ingredients_name', 'prep_time (in mins)', 'cook_time (in mins)', 'description']].iloc[food_indices]
        result.reset_index(drop=True, inplace=True)
        return result


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
    # st.dataframe(indices)

    recommend_food_general(select_food, cosine_sim)


    tf_vect = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1,3), 
                          stop_words='english', strip_accents='unicode')

    result['description'] = result['description'].fillna('')

    desc_matric = tf_vect.fit_transform(result['description'])

    # st.text('we peek through the TF-IDF shape')
    # st.write(desc_matric.shape)

    # st.write('Since we have converted the series to a matrix, we use the Sigmoid Kernel to compute the metrics pairwise of the content and prediction (X, Y)')
    sig = sigmoid_kernel(desc_matric, desc_matric)

    # Create a new dataframe holding the movie titles and index series
    indices = pd.Series(result.index, result['name']).drop_duplicates()


    # def fetch_movie_image(movie_title):
    #     api_key = os.getenv("api_key")
    #     base_url = 'http://www.omdbapi.com/?apikey=' + api_key + '&t='
    #     response = requests.get(base_url + movie_title)
    #     movie_data = response.json()

    #     if 'Poster' in movie_data:
    #         return movie_data['Poster']
    #     else:
    #         return 'https://via.placeholder.com/200x300.png?text=No+Image+Available'


    def fetch_food_image(food_name):
        search_url = f"https://www.google.com/search?hl=en&q={food_name.replace(' ', '+')}&tbm=isch"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        image_tags = soup.find_all("img")
        
        if len(image_tags) > 1:
            return image_tags[1]["src"] 
        else:
            return image_tags[1]["https://via.placeholder.com/200x300.png?text=No+Image+Available"]
            


    def recommend_food_narrative(meal, sig=sig):
    
        food_list = [x for x in result['name']]
        
        if meal in food_list:
        
            idx = indices[meal]

            food_list = list(enumerate(sig[idx]))

            sort_food = sorted(food_list, key = lambda x: x[1], reverse = True)

            top_ten = sort_food[1:21]

            food_rec = [x[0] for x in top_ten]
            
            st.text("")
            
            st.text('Food For Thought')

            st.dataframe(pd.DataFrame(result[['name', 'cuisine', 'course', 'diet', 'prep_time (in mins)', 'cook_time (in mins)', 'ingredients_name']].iloc[food_rec]))

            # for i in range(0, len(food_rec), 3):
            #     row = food_rec[i:i + 3]
            #     with st.expander("", expanded=True):
            #         st.image(fetch_food_image(df['name'][row[0]]), width=200)
            #         st.image(fetch_food_image(df['name'][row[1]]), width=200)
            #         st.image(fetch_food_image(df['name'][row[2]]), width=200)
            #     st.write("")

            for i in range(0, len(food_rec), 3):
                
                col1, col2, col3 = st.columns(3)
                row = food_rec[i:i + 3]

                with col1:
                    if len(row)>=1:
                        st.image(fetch_food_image(result['name'][row[0]]), caption=result['name'][row[0]], use_container_width=True, channels='BGR')
                with col2:
                    if len(row)>=2:
                        st.image(fetch_food_image(result['name'][row[1]]), caption=result['name'][row[1]], use_container_width=True, channels='BGR')
                with col3:
                    if len(row)>=3:
                        st.image(fetch_food_image(result['name'][row[2]]), caption=result['name'][row[2]], use_container_width=True, channels='BGR')


            # for i in food_rec:
            #     meal_name = df.iloc[i]['name']
            #     meal_img_url = fetch_food_image(meal_name)
            #     st.image(meal_img_url, width=200)

            # return st.write(result[['name', 'cuisine', 'course', 'diet', 'prep_time (in mins)', 'cook_time (in mins)', 'ingredients_name']].iloc[food_rec].sort_values(by = ["course", "cuisine"], ascending = False))
        
        else:
            st.text('__DATABASE ERROR___ During handling of the above exception, another exception occurred: InvalidIndexError(key)')
            st.text('MOVIE NOT FOUND')


    # st.image("Build-a-Recommendation-Engine-With-Collaborative-Filtering_Watermarked.webp")
    
    recommend_food_narrative(result['name'].iloc()[0])
    # netflix_recommender(select_movie)
    
    with st.spinner("Loading..."):
        time.sleep(5)
        st.success("Done!")


    st.markdown('credits EchoMinds Innovation [(www.echominds.africa)]')
                                                                         


#########################################################################################################



# from IPython.display import display, Image

# Function to get and display food images
# def show_food_image(food_name):
#     search_url = f"https://www.google.com/search?hl=en&q={food_name.replace(' ', '+')}&tbm=isch"
#     headers = {"User-Agent": "Mozilla/5.0"}

#     # Request Google Images
#     response = requests.get(search_url, headers=headers)
#     soup = BeautifulSoup(response.text, "html.parser")

#     # Extract first image URL
#     image_tags = soup.find_all("img")
#     if len(image_tags) > 1:
#         img_url = image_tags[1]["src"]
#         print(f"Image for {food_name}: {img_url}")
        
#         # Display image (Works in Jupyter Notebook)
#         # display(Image(url=img_url))
#         st.image(img_url)
#     else:
#         print(f"No image found for {food_name}")

# # Example usage
# food_list = ["Jollof Rice", "Samosa", "Injera", "Chapati", "Pounded Yam"]
# for food in food_list:
#     show_food_image(food)




###############################################################################################

# import requests

# def find_osm_restaurants(food_name, lat=1.2921, lon=36.8219, radius=5000):
#     overpass_url = "http://overpass-api.de/api/interpreter"
#     query = f"""
#     [out:json];
#     node["amenity"="restaurant"](around:{radius},{lat},{lon});
#     out;
#     """
    
#     response = requests.get(overpass_url, params={"data": query})
#     data = response.json()
    
#     restaurants = []
#     for element in data.get("elements", []):
#         name = element.get("tags", {}).get("name", "Unknown Restaurant")
#         restaurants.append(name)
    
#     return restaurants[:5]  # Return top 5 restaurants

# # Example usage
# food_recommendations = ["Jollof Rice", "Samosa", "Chapati"]
# for food in food_recommendations:
#     st.write(f"Restaurants for {food}: {find_osm_restaurants(food)}")


