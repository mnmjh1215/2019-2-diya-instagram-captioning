from sklearn.metrics import f1_score


yt = ["1212", "테스트용 입니다"]
yp = ["1313", "테스트용"]
def f1(y_true, y_pred):
    
    return f1_score(y_true, y_pred, average = 'binary')

f1(yt, yp)