# demo web application

from flask import Flask, render_template, request, redirect, session, url_for

from konlpy.tag import Okt
import jpype

import json
import torch


app = Flask(__name__)

HASHTAG_VOCAB_PATH = ""
HASHTAG_MODEL_PATH = ""

TEXT_VOCAB_PATH = ""
TEXT_MODEL_PATH = ""

@app.route("/")
def index():
    return "Hello DIYA!"

# TODO: 업로드한 파일로 inference 실행 및 결과 리턴



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)




