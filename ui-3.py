import pickle
import streamlit as st
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import re
conn = mysql.connector.connect(

    host="localhost",

    user="root",

    port="3306",

    password="seenu2218",

    database="finalproject"

)

table_name='youtube'
database="finalproject"
cursor = conn.cursor()

writer = cursor 

query = "SELECT * FROM youtube"
cursor.execute(query)
view=cursor.fetchall()

query2 =f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = '{database}' ORDER BY ORDINAL_POSITION"
cursor.execute(query2)
a=cursor.fetchall()
flat_list = [item[0] for item in a]
final=pd.DataFrame(view)
final.columns=flat_list
final["statistics.viewCount"]=final["statistics.viewCount"].fillna(0)
final["statistics.viewCount"]=final["statistics.viewCount"].astype(int)
final["statistics.likeCount"]=final["statistics.likeCount"].fillna(0)
final["statistics.likeCount"]=final["statistics.likeCount"].astype(int)
with open('model.pkl',"rb") as f:
    kmeans=pickle.load(f)
with open('vectorizer.pkl',"rb") as k:
    vectorizer=pickle.load(k)  
with open('X.pkl',"rb") as m:
    X=pickle.load(m)       
def recommend_channels(query):
    query_vector = vectorizer.transform([query])
    input_cluster = kmeans.predict(query_vector)[0]
    cluster_channels = final[final['cluster'] == input_cluster]
    similarity_scores = cosine_similarity(query_vector, X[cluster_channels.index]).flatten()
    recommended_indices = similarity_scores.argsort()[::-1]
    recommended_channels = cluster_channels.iloc[recommended_indices]
    return recommended_channels
log,sig=st.columns([5,1])
hello=False
with sig:
      with st.popover("sign in "):
          a=st.text_input('yourname')
          if a:
            mat=re.match(r'[A-z a-z 0-9]+',a)
            if mat:
                print('success')
            else:
                mail=False
                st.error('please enter name') 
          b=st.text_input('your mail id')
          mail=True
              
          if b:
            pattern=re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",b)
            if pattern:
                print('success')
            else:
                mail=False
                c=st.error('invalid mail address')
          c=st.text_input('age')
          if c:
            num=re.match('[0-9]+',c)
            if num:
                print('success')
            else:
                mail=False
                mun=st.error('invalid age')             
          button=st.button('submit')
          if a and button and mail:
                hello=True
                st.success('signed in successfully')
                cursor.execute('insert into signin(`name`,`mailid`,`age`) VALUES (%s, %s, %s)',(a,b,int(c)))
                conn.commit()
with log:
    if hello: 
        st.write(f"Hello :red[{a}]!!")
        st.balloons()                

m,n=st.columns([1,10])
with m:
    st.image(r"1000_F_300389025_b5hgHpjDprTySl8loTqJRMipySb1rO0I.jpg")
#a,b=st.columns([1,10])
#with b:

st.sidebar.image(r"E:\guvi final\2nd.jpg")
a=[None,'highestviews','highestlikes']
c = st.sidebar.radio("Trending🔥", options=a)
with st.sidebar.expander('Select channel'):
    fin=[i for i in final['snippet.channelTitle'].drop_duplicates()]
    select_channel=st.radio("Select Channel",fin)     
query=st.text_input('please enter')
def x(final4):
    for a,b in final4.iterrows():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(b['snippet.thumbnails.default.url'], use_column_width=True)
        with col2:
            st.write(f'__{b['snippet.title']}__')
            st.write(f':red[{b['snippet.channelTitle']}]')
            k,h=st.columns([1,3])
            with k:
                st.write(":blue[View's]",int(b['statistics.viewCount']))
            with h:
                st.write('👍',b['statistics.likeCount'])
        st.divider()
check=True           
if query:
#with b:
    check=False
    o=recommend_channels(query)
    col1,col2=st.columns([1,2])
    x(o)        

else:
    if c=='highestviews':
        final2 = final.sort_values(by='statistics.viewCount', ascending=False)
        final2=final2.iloc[:10]
        x(final2)
    elif c=='highestlikes':
        final3=final.sort_values(by='statistics.likeCount',ascending=False)
        final3=final3.iloc[:10]
        x(final3)

if c is None and select_channel and check:
    final4=final[final['snippet.channelTitle']==select_channel]
    x(final4)

    