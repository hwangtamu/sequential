from src.LSTM import *

c = DNALSTM()
# c.train(save_model=True)
p="./saved_model/lstm-dna_128.h5"
# c.evaluate(model=p)
c.real_time_predict(model=p, sample=200, padding=500)
