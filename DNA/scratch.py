from src.LSTM import *

for i in [4,6,8,10,12,16,32,64,128]:
    print('='*30+str(i)+'='*30)
    p="./saved_model/lstm-dna_b_3_"+str(i)+".h5"
    c = DNALSTM(i, 3)
    # c.train(save_model=True)
    # c.evaluate(model=p)
    c.visualize(model=p, sample=30, padding=500)
