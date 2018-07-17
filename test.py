from LSTM import MnistLSTMClassifier

lstm_classifier = MnistLSTMClassifier()
#lstm_classifier.visualize(model="./saved_model/lstm-model.h5", samples=100)
lstm_classifier.vis_hidden(model="./saved_model/lstm-model.h5", samples=200)