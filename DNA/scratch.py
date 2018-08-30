from src.LSTM import *

for j in range(30):
    for i in [4,6,8,10,12,16,32,64,128]:
        print('='*30+str(i)+'='*30)
        p="./saved_model/lstm-dna_"+str(j)+"_3_"+str(i)+".h5"
        c = DNALSTM(i, 3, str(j))
        # c.train(save_model=True)
        # c.evaluate(model=p)
        c.visualize(model=p, sample=30, padding=500)
