import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def bleu(actuals, predictions, n=1):
    """
    :param actuals: list of list of actual targets (texts or hashtags), <start>, <end> 제거
    :param predictions: list of list of predicted targets, <start>, <end> 제거
    :param avg: average method
    :return: macro_averaged bleu score
    """
    
    weights = tuple([1/n] * n)

    scores = []
    for actual, pred in zip(actuals, predictions):
        # actual과 pred 중 n보다 길이가 더 작은 문장이 있다면, 스킵
        if min(len(actual), len(pred)) < n:
            continue
        score = sentence_bleu([actual], pred, weights=weights)
        scores.append(score)
    return sum(scores) / len(scores)
        


def precision(actual, prediction, start_token_index=2, end_token_index=3):
    """
    actual: 실제 해시태그 목록, <start>와 <end> 제외
    prediction: 예측 해시태그 목록, <start>와 <end> 제외
    """
    
    actual = set(actual)
    prediction = set(prediction)
    
    actual.discard('<start>')
    actual.discard('<end>')
    actual.discard(start_token_index)
    actual.discard(end_token_index)
    
    prediction.discard('<start>')
    prediction.discard('<end>')
    prediction.discard(start_token_index)
    prediction.discard(end_token_index)
    
    return len(actual.intersection(prediction)) / len(prediction)


def recall(actual, prediction, start_token_index=2, end_token_index=3):
    """
    actual: 실제 해시태그 목록, <start>와 <end> 제외
    prediction: 예측 해시태그 목록, <start>와 <end> 제외
    """
    
    actual = set(actual)
    prediction = set(prediction)
    
    actual.discard('<start>')
    actual.discard('<end>')
    actual.discard(start_token_index)
    actual.discard(end_token_index)
    
    prediction.discard('<start>')
    prediction.discard('<end>')
    prediction.discard(start_token_index)
    prediction.discard(end_token_index)
    
    return len(actual.intersection(prediction)) / len(actual)


def f1_score(actual, prediction):
    prec = precision(actual, prediction)
    rec = recall(actual, prediction)
    f1 = 2 * ((prec * rec)/(prec + rec))
    return f1

