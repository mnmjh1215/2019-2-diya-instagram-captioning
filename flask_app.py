# demo web application

from flask import Flask, render_template, request, redirect, session, url_for
import json
import torch
from utils import OktDetokenizer
from konlpy.tag import Okt
import jpype

app = Flask(__name__)

okt = Okt()
detok = OktDetokenizer()

HASHTAG_VOCAB_PATH = ""
HASHTAG_MODEL_PATH = ""

TEXT_VOCAB_PATH = ""
TEXT_MODEL_PATH = ""

@app.route("/")
def index():
    return render_template('instagram_template.html')


@app.route('/<text>')
def tokenize_and_detokenize(text):
    jpype.attachThreadToJVM()

    tokenized_text = okt.pos(text, norm=True, join=True)
    detokenized_text = detok.detokenize(tokenized_text)
    
    return "Tokenization result: {0} <br /> Detokenization result: {1}".format(" ".join(tokenized_text), detokenized_text)

    
# TODO: 업로드한 파일로 inference 실행 및 결과 리턴



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)




