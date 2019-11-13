# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:38:32 2019

@author: 윤준석
"""
import os
import re
import json 


#광고 구분할 키워드
ad_keyword = ["광고", "문의", "연락", "상담"]


#한글 해시태그만 찾기
def get_parsed_hashtags(raw_text):
    hashtags = re.findall(r'#([ㄱ-ㅣ가-힣]+)', raw_text) 
    return hashtags


#광고 구분
def is_advertise(text):
    for i in ad_keyword:
        if i in text:
            return True
    return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Merge all json files in given directory")
    
    parser.add_argument('--dir',
                        default='./')
    
    args = parser.parse_args()
    
    files = os.listdir(args.dir)

    posts = []

    for f in files:
        if not f.endswith('.json'):
            # json 파일이 아닌 파일의 경우 스킵
            continue
        
        with open(os.path.join(args.dir, f), 'rt', encoding='UTF8') as json_file:
            file = json.load(json_file)
            
            key_set = set()
            
            for instance in file:
                if instance['key'] not in key_set and 'description' in instance: #동일한 게시물 제거
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
    with open(os.path.join(args.dir, 'preprocess.json'), 'w', encoding='utf-8') as make_file:
        json.dump(posts, make_file, ensure_ascii=False)
