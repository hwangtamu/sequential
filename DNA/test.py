from padded import MnistLSTMClassifier
from utils import Data
from keras.models import load_model

# lstm_classifier.visualize(model="./saved_model/lstm-model_32.h5", num=3, samples=100, permute=True)
# lstm_classifier.vis_hidden(model="./saved_model/lstm-model_32.h5", num=2, samples=100, permute=False, padding=200)


def real_time_pred(model=None, num=None, sample=100, padding=None, mc=False):
    lstm_classifier = MnistLSTMClassifier()
    d = Data()
    if num!=None:
        hidden_states, results, ids = lstm_classifier.get_hidden(model=model,
                                                                 num=num,
                                                                 samples=sample,
                                                                 padding=padding,
                                                                 mc=mc)
        for i in range(len(hidden_states)):
            for t in hidden_states[i]:
                pred, tmp, tmp_o = lstm_classifier.real_time_predict(t)
                print(ids[i], num, pred, [(tmp[x], float('%.3f'%(tmp_o[x]))) for x in range(padding) if x % 5 == 0])
    else:
        for n in range(3):
            hidden_states, results, ids = lstm_classifier.get_hidden(model=model,
                                                                     num=n,
                                                                     samples=sample,
                                                                     padding=padding,
                                                                     mc=mc)
            for i in range(len(hidden_states)):
                for t in hidden_states[i]:
                    pred, tmp, tmp_o = lstm_classifier.real_time_predict(t)
                    print(ids[i], n, pred, [(tmp[x], float('%.3f'%(tmp_o[x]))) for x in range(padding) if x%5 == 0])
                    # if n!=pred:
                    #     d.visualize_by_id(ids[i])


def sort_by_confidence(model=None, num=None, sample=100, padding=None, test=True):
    lstm_classifier = MnistLSTMClassifier()
    res = []
    if num != None:
        hidden_states, results, ids = lstm_classifier.get_hidden(model=model, num=num, samples=sample, padding=padding, test=test)
        for i in range(len(hidden_states)):
            #if results[i]==1:
            for t in hidden_states[i]:
                pred, tmp, tmp_o = lstm_classifier.real_time_predict(t, num)
                res+=[(ids[i], tmp_o[28])]
        return sorted(res, key=lambda x: -x[1])


if __name__=="__main__":
    model = "./saved_model/lstm-model_128.h5"
    lstm_classifier = MnistLSTMClassifier()
    # lstm_classifier.visualize(model, samples=1000)
    real_time_pred(model=model, sample=100, padding=200, mc=False)
    # l = sort_by_confidence(model=model, num=5, sample=1000, padding=150, test=True)
    # ids, params = zip(*l)
    # d = Data()
    # d.serial_viz_by_id(ids,params, test=True)

