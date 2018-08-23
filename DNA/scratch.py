from src.LSTM import *

for i in [4,6,8,10,12,16,32,64,128]:
    p="./saved_model/lstm-dna_"+str(i)+".h5"
    c = DNALSTM(i)
    #c.train(save_model=True)
    c.evaluate(model=p)
    # c.real_time_predict(model=p, sample=307, padding=61)
