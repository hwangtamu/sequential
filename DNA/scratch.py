from src.LSTM import MnistLSTMClassifier

c = MnistLSTMClassifier()
p="./saved_model/lstm-model_32.h5"
c.real_time_predict(model=p, padding=150)
