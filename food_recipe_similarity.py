#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# In[2]:


import pandas as pd
import numpy as np

food = pd.read_csv("new_df.csv")
food


# In[3]:


list(food['메뉴설명(MENU_DSCRN)'])


# ## 메뉴재료로만 유사도 계산
# #### BOW 방법 : 전체 문서를 구성하는 고정된 단어장을 만들고 di라는 개별 문서에 단어장에 해당하는 단어들이 포함되어 있는지를 표시
# 
# ##### xi,j = 문서 di내의 단어 tj의 출현 빈도 (만약 단어 tj가 문서 di 안에 없으면 0, 단어 tj가 문서 di 안에 있으면 1
# 
# ###### TfidfVectorizer : CountVectorizer와 비슷하지만 TF-IDF 방식으로 단어의 가중치를 조정한 BOW 벡터를 만든다. -> TF-IDF(Term Frequency – Inverse Document Frequency) 인코딩은 단어를 갯수 그대로 카운트하지 않고 모든 문서에 공통적으로 들어있는 단어의 경우 문서 구별 능력이 떨어진다고 보아 가중치를 축소하는 방법이다.

# In[4]:


# n-그램 : 단어장 생성에 사용할 토큰의 크기를 겾렁.
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0.0)
tfidf_matrix = tf.fit_transform(food['메뉴설명(MENU_DSCRN)'])


# In[5]:


print(tfidf_matrix[10])


# In[6]:


tfidf_matrix.shape


# ##### 코사인 유사도를 사용해 유사성 계산

# In[7]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[8]:


food_cates = food['메뉴카테고리소분류명']
indices = pd.Series(food.index, index=food['메뉴카테고리소분류명'])

print(food_cates.head(), indices.head())


# In[9]:


def get_recommendations(food_cate):
    idx = indices[food_cate]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    food_indices = [i[0] for i in sim_scores]
    return food_cates.iloc[food_indices][:5]


# In[10]:


get_recommendations('치킨')


# In[11]:


display(food[food['메뉴카테고리소분류명']=='추어탕'])
display(food[food['메뉴카테고리소분류명']=='주먹밥'])


# ## 대분류명, 소분류명 이용하여 유사도 계산

# In[12]:


food['soup'] = food['메뉴설명(MENU_DSCRN)'] + "  " + food['메뉴카테고리대분류명'] + "  " + food['메뉴카테고리소분류명']
food['soup']


# In[13]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), min_df=0.0, stop_words='english')
tfidf_matrix = tf.fit_transform(food['soup'])


# In[14]:


sorted(tf.vocabulary_.items())


# In[15]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim


# In[16]:


food_cates = food['메뉴카테고리소분류명']
indices = pd.Series(food.index, index=food['메뉴카테고리소분류명'])


# In[17]:


get_recommendations('소갈비').head(3)


# In[18]:


display(food[food['메뉴카테고리소분류명']=='갈비'])
display(food[food['메뉴카테고리소분류명']=='갈비찜'])
display(food[food['메뉴카테고리소분류명']=='스테이크'])
display(food[food['메뉴카테고리소분류명']=='생굴'])


# In[19]:


get_recommendations('닭갈비').head(3)


# In[20]:


get_recommendations('치즈돈까스').head(5)


# In[21]:


food_simi_co = cosine_sim * 1


# In[22]:


food_simi_co_sorted_ind = food_simi_co.argsort()[:, ::-1]

def find_simi_food(food_name):
    
    idx = indices[food_name]
    sim_scores = list(enumerate(food_simi_co[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    food_indices = [i[0] for i in sim_scores]
    return food_cates.iloc[food_indices][:5]


# ## 카테고리 가중치 추가

# In[23]:


find_simi_food('닭갈비')


# In[24]:


menu_simi_co = (cosine_sim * 1)
menu_simi_co_sorted_ind = food_simi_co.argsort()[:, ::-1]

# 최종 구현 함수
def find_simi_menu(food_name):
    
    idx = indices[food_name]    
    
    #사용자에 대한 메뉴 가중치 파일 받아오기
    food_ratio = pd.read_csv('./food_ratio.csv', encoding='cp949', index_col=0)
    
    a = list(food['메뉴카테고리대분류명'])
    replacements = {'한식':float(food_ratio.loc['한식']),
                   '제과류':float(food_ratio.loc['카페']),
                   '양식':float(food_ratio.loc['양식']),
                   '아시아/퓨전 음식':float(food_ratio.loc['아시아/퓨전 음식']),
                   '일식':float(food_ratio.loc['일식']),
                   '패스트푸드':float(food_ratio.loc['분식/치킨']),
                   '중식':float(food_ratio.loc['중식'])}
    replacer = replacements.get
    
    w = 1 #카테고리 가중치 설정
    #카테고리 선호비율을 축소하고 최종 추천된 메뉴점수에 더하기
    food_simi = food_simi_co[idx]+ [x*y for x,y in zip([replacer(n, n) for n in a],[w]*len(a))]
    sim_scores = list(enumerate(food_simi))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    food_indices = [i[0] for i in sim_scores]
    return food_cates.iloc[food_indices][:5]


# ## 사용자유형 가중치 추가

# In[25]:


food_simi_co = (cosine_sim * 1)

food_simi_co_sorted_ind = food_simi_co.argsort()[:, ::-1]

# 입력변수가 하나 더 생겨서 파일 한꺼번에 돌릴시 중간에러를 방지하기 위해 함수이름을 바꿉니다..
def find_simi_food_ver3(food_name,w): #여기서 w는 사용자유형에 대한 가중치
    
    idx = indices[food_name]
    
    #사용자에 대한 메뉴 가중치 파일 받아오기
    food_ratio = pd.read_csv('./food_ratio.csv', encoding='cp949', index_col=0)
    
    a = list(food['메뉴카테고리대분류명'])
    replacements = {'한식':float(food_ratio.loc['한식']),
                   '제과류':float(food_ratio.loc['카페']),
                   '양식':float(food_ratio.loc['양식']),
                   '아시아/퓨전 음식':float(food_ratio.loc['아시아/퓨전 음식']),
                   '일식':float(food_ratio.loc['일식']),
                   '패스트푸드':float(food_ratio.loc['분식/치킨']),
                   '중식':float(food_ratio.loc['중식'])}
    replacer = replacements.get
    cate_w = 1 #카테고리 가중치 설정
    
    # 사용자 유형의 메뉴선호 반영
    user_type_like = pd.read_csv('./user_type_like', index_col=0)
    
    food_simi = food_simi_co[idx]+ [x*y for x,y in zip([replacer(n, n) for n in a],[cate_w]*len(a))] + user_type_like.loc[w[0]].values *(0.01)*w[1]
    sim_scores = list(enumerate(food_simi))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    food_indices = [i[0] for i in sim_scores]
    return food_cates.iloc[food_indices][:5]


# In[26]:


#확인용
find_simi_food_ver3('닭갈비',['건강식단추구',0])


# In[ ]:


food_simi_co


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
transformer = MinMaxScaler()
transformer.fit(food_simi_co)
simi_food = transformer.transform(food_simi_co)
simi_food = pd.DataFrame(simi_food)
simi_food


# In[ ]:


simi_food.to_csv('simi_food')


# In[ ]:




