import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from tqdm import tqdm

def avg_bleu(actuals, predictions, n=1):
    """
    :param actuals: list of list of actual targets (texts or hashtags), <start>, <end> 제거
    :param predictions: list of list of predicted targets, <start>, <end> 제거
    :param avg: average method
    :return: macro_averaged bleu score
    """
    
    weights = tuple([1/n] * n)

    scores = []
    for actual, pred in tqdm(zip(actuals, predictions)):
        # actual과 pred 중 n보다 길이가 더 작은 문장이 있다면, 스킵
        if len(actual) < n:
            continue
        elif len(pred) < n:
            scores.append(0)
        else:
            score = sentence_bleu([actual], pred, weights=weights)
            scores.append(score)
            
    return sum(scores) / len(scores)
        
        
def get_lcs(X , Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in range(m+1)] 
  
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n]       


def rouge_l(actual, pred, beta=8):
    lcs = get_lcs(actual, pred)
    prec_l = lcs / len(pred) if len(pred) != 0 else 0
    recall_l = lcs / len(actual) if len(actual) != 0 else 0
    
    if (recall_l + prec_l) == 0:
        return 0
    rouge_l_score = (1 + beta ** 2) * recall_l * prec_l / (recall_l + beta ** 2 * prec_l)
    return rouge_l_score
    

def avg_rouge_l(actuals, preds, beta=8):
    scores = []
    for actual, pred in tqdm(zip(actuals, preds)):
        rouge_l_score = rouge_l(actual, pred, beta=beta)
        scores.append(rouge_l_score)
        
    return sum(scores) / len(scores)


def avg_meteor(actuals, preds):
    scores = []
    for actual, pred in tqdm(zip(actuals, preds)):
        actual = " ".join(str(idx) for idx in actual)
        pred = " ".join(str(idx) for idx in pred)
        score = single_meteor_score(actual, pred)
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
    
    correct = len(actual.intersection(prediction))
    denominator = len(prediction)
    
    if denominator == 0:
        return 0, correct, denominator
    
    return correct / denominator, correct, denominator


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
    
    correct = len(actual.intersection(prediction))
    denominator = len(actual)
        
    if denominator == 0:
        return 0, correct, denominator
    return correct / denominator, correct, denominator


def f1_score(actual, prediction):
    prec = precision(actual, prediction)
    rec = recall(actual, prediction)
    if (prec + rec) == 0:
        return 0
    f1 = 2 * ((prec * rec)/(prec + rec))
    return f1


def avg_f1_score(actuals, predictions):
    total_correct = 0
    total_precision_denominator = 0
    total_recall_denominator = 0
    for actual, pred in zip(actuals, predictions):
        prec, correct, prec_denominator = precision(actual, pred)
        rec, correct, rec_denominator = recall(actual, pred)
        total_correct += correct
        total_precision_denominator += prec_denominator
        total_recall_denominator += rec_denominator
    prec = total_correct / total_precision_denominator
    rec = total_correct / total_recall_denominator
    
    avg_f1 = 2 * ((prec * rec)/(prec + rec))
    return avg_f1, prec, rec
