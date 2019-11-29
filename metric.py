import numpy as np





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

