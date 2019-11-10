# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:38:32 2019

@author: 윤준석
"""
import os
import re
import json 

#raw_file directory
dir = "C:\\Users\\윤준석\\Desktop\\instagram-crawler-master\\instagram-crawler-master\\"

files = os.listdir(dir)

posts = []

#광고 구분할 키워드
ad_keyword = ["광고", "문의", "연락", "상담"]


#한글 해시태그만 찾기
def get_parsed_hashtags(raw_text):
    hangeul_regex = re.compile(r'#([^ ㄱ-ㅣ가-힣]+)')
    hangeul = hangeul_regex.sub("", raw_text)
    
    hashtag_regex = re.compile(r'#(\w+)')
    hashtags = hashtag_regex.findall(hangeul)
    return hashtags

#광고 구분
def is_advertise(text):
    for i in ad_keyword:
        if i in text:
            return True
    return False


for f in files:
    with open(dir+f, 'rt', encoding='UTF8') as json_file:
        file = json.load(json_file)
        
        key_set = set()
        
        for instance in file:
        
            if instance['key'] not in key_set: #동일한 게시물 제거
                if not is_advertise(instance['description']): #광고 제거
                    hashtags = get_parsed_hashtags(instance['description']) 
                    #해시태그 없는 게시물 제거
                    if hashtags != []:
                        dict_post = {'key' : instance['key']}
                        dict_post['img_url'] = instance['img_url']
                        dict_post['hashtag'] = hashtags
                        print(hashtags)
                        posts.append(dict_post)
                        key_set.add(instance['key'])
#하나의 파일로 출력
with open(dir+'preprocess.json', 'w', encoding='utf-8') as make_file:

    json.dump(posts, make_file, ensure_ascii=False)