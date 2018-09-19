from src.LSTM import *

# r = ReduceMNIST()
#
# x = r.x_test
# y = r.y_test
a = [0 for _ in range(9)]
b = [4,6,8,10,12,16,32,64,128]

for j in range(30):
    for i in range(len(b)):
        print('='*30+str(b[i])+'='*30)
        p="./saved_model/lstm-dna_"+str(j)+"_3_"+str(b[i])+".h5"
        # p = "./saved_model/lstm-dna_0_3"+str(b[i])+".h5"
        c = DNALSTM(b[i],3,str(j))
        # c = DNALSTM(b[i], n_classes=3, indicator=str(j))
        # c.train(save_model=True)
        # c.real_time_predict(model=p, sample=20,padding=100)
        c.get_states(model=p, sample=20, padding=100)
        # c.visualize(model=p, sample=100, padding=500)
        # visualize(x,y,28,i,28,model=p, padding=100)

