from src.LSTM import *

a = [0 for _ in range(9)]
b = [4,6,8,10,12,16,32,64,128]

#for j in range(30):
    # print('='*30+str(b[i])+'='*30)
    #p="./saved_model/gru-dna-mse_"+str(j)+"_3_32.h5"
    # p = "./saved_model/lstm-dna_0_3"+str(b[i])+".h5"
    #c = DNAGRU(32,3,str(j))
    # c = DNALSTM(b[i], n_classes=3, indicator=str(j))
    # c.train(save_model=True)
    # c.real_time_predict(model=p, sample=20,padding=100)
    # c.dense_dots(model=p, sample=50, padding=60)
    # c.visualize(model=p, sample=100, padding=500)
    # visualize(x,y,28,i,28,model=p, padding=100)
    # c.lesion_eval(model=p, acuity=38)
    #c.act_vis(model=p)


p="./saved_model/lstm-dna-mse_"+'0'+"_3_128.h5"
c = DNALSTM(128,3,'0')
c.get_corr(model=p)
# c.lesion_eval(model=p, acuity=1, n=[2, 68, 80, 82, 97, 115, 96, 93, 81, 64, 103])
# for i in range(128):
#     c.lesion_eval(model=p, acuity=1, n=i)
# for i in range(300):
#     c.act_vis(model=p, id=i)
