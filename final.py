import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from sqlalchemy import create_engine
with open("food.json","r") as file :
  a=json.load(file)
with open("data.json","r") as file :
  b=json.load(file)
with open("news.json","r") as file :
  c=json.load(file)
with open("sports.json","r") as file :
  d=json.load(file)
with open("educational.json","r") as file :
  e=json.load(file)   
food=[]
for i in a:
    if "items" in i:
        food.extend(i['items'])
data=[]
for i in b:
    if "items" in i:
        data.extend(i['items'])
news=[]
for i in c:
    if "items" in i:
        news.extend(i['items'])
sports=[]
for i in d:
    if "items" in i:
        sports.extend(i['items'])
educational=[]
for i in e:
    if "items" in i:
        educational.extend(i['items'])
df1=pd.json_normalize(food)
df2=pd.json_normalize(data)
df3=pd.json_normalize(news)
df4=pd.json_normalize(sports)
df5=pd.json_normalize(educational)
final=pd.concat([df1,df2,df3,df4,df5])
final=final.dropna(subset=['snippet.tags'])
nltk.download("stopwords")
c = set(stopwords.words('english'))
final['snippet.tags'] = final['snippet.tags'].apply(lambda x: [word for word in x if word not in c])
lemmatizer= WordNetLemmatizer()
nltk.download('wordnet')
final['snippet.tags'] = final['snippet.tags'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x if word not in c])
final["snippet.tags"]=final["snippet.tags"].apply(lambda x: " ".join(x) if isinstance(x,list) else None)
load_dotenv()



embedding = HuggingFaceHubEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)


embeddings = embedding.embed_documents(final["snippet.tags"])
embeddings = np.array(embeddings) 

# Cluster the embeddings
num_clusters = 100  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
final['cluster'] = kmeans.fit_predict(embeddings)  

def recommend_channels(query):
    query_vector = np.array(embedding.embed_query(query)).reshape(1, -1)

    input_cluster = kmeans.predict(query_vector)[0]
    cluster_channels = final[final['cluster'] == input_cluster]

    selected_embeddings = embeddings[cluster_channels.index.to_numpy()]

    similarity_scores = cosine_similarity(query_vector, selected_embeddings).flatten()
    recommended_indices = similarity_scores.argsort()[::-1]
    recommended_channels = cluster_channels.iloc[recommended_indices]

    print("Recommended channels:")
    print(recommended_channels)

query = input("Search relevant content: ")
if query:
    recommend_channels(query)

engine=create_engine('sqlite:///youtube.db')
final.to_sql('youtube',engine,index=False)

