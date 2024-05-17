#!/usr/bin/env python
# coding: utf-8

# In[205]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

cats = ['놀람', '공포', '분노', '슬픔', '중립', '혐오', '행복']

content_df = pd.read_excel("감정_데이터셋.xlsx")
content_df


# In[206]:


new_df = content_df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', '공포', 5468], axis=1)


# In[207]:


new_df


# In[304]:


index = 38594
new_df.loc[index] = ['난생 처음으로 해커톤을 나갔다. 처음이라 너무 힘들기도 했고 내가 팀에 아무런 도움도 되지 않아서 너무 슬펐다. 다음에는 꼭 팀에 민폐가 되지 않아야겠다고 다짐했다, 그래도 한편으로는 새로운 경험을 한 것 같아 뿌듯했다', None]
index+=1
print(new_df.loc[index-1])


# In[305]:


new_df


# In[301]:


f = open('불용어.txt', "r", encoding='utf-8')
data = f.read().split()
print(data)


# In[302]:


text_data = content_df['Sentence'].dropna()


#LDA 는 Count기반의 Vectorizer만 적용  
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words=data, ngram_range=(1,2))
feat_vect = count_vect.fit_transform(text_data)
print('CountVectorizer Shape:', feat_vect.shape)


# In[255]:


count_vect


# In[212]:


feat_vect


# In[224]:


lda = LatentDirichletAllocation(n_components=7, random_state=0)
lda.fit(feat_vect)


# **각 토픽 모델링 주제별 단어들의 연관도 확인**  
# lda객체의 components_ 속성은 주제별로 개별 단어들의 연관도 정규화 숫자가 들어있음
# 
# shape는 주제 개수 X 피처 단어 개수  
# 
# components_ 에 들어 있는 숫자값은 각 주제별로 단어가 나타난 횟수를 정규화 하여 나타냄.   
# 
# 숫자가 클 수록 토픽에서 단어가 차지하는 비중이 높음  

# In[271]:


print(lda.components_.shape)
lda.components_


# **각 토픽별 중심 단어 확인**

# In[303]:


def display_topic_words(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('\nTopic #',topic_index)

        # components_ array에서 가장 값이 큰 순으로 정렬했을 때, 그 값의 array index를 반환. 
        topic_word_indexes = topic.argsort()[::-1]
        top_indexes=topic_word_indexes[:no_top_words]
        
        # top_indexes대상인 index별로 feature_names에 해당하는 word feature 추출 후 join으로 concat
        feature_concat = ' '.join([str(feature_names[i]) for i in top_indexes])
        #feature_concat = ' + '.join([str(feature_names[i])+'*'+str(round(topic[i],1)) for i in top_indexes])                
        print(feature_concat)

# CountVectorizer객체내의 전체 word들의 명칭을 get_features_names( )를 통해 추출
feature_names = count_vect.get_feature_names_out()

# Topic별 가장 연관도가 높은 word를 15개만 추출
display_topic_words(lda, feature_names, 5)


# **개별 문서별 토픽 분포 확인**
# 
# lda객체의 transform()을 수행하면 개별 문서별 토픽 분포를 반환함. 

# In[306]:


doc_topics = lda.transform(feat_vect)
print(doc_topics.shape)
print(doc_topics[-1])


# In[ ]:




