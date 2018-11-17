import numpy as np
from numpy import linalg
d = 3
data_n = 4
ita = 0.1

w = np.random.rand(d,1)
x = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]], dtype = 'f')
x = x.T
r = np.array([0, 0, 0, 1], dtype = 'f')
y = np.zeros((data_n,1))
g = np.array([[1],[1],[1]])

for epoch in range(500):    
    del_w = np.zeros((d,1))     
    for t in range(data_n):   
        o = w.T.dot(x[:,t])
        y[t] = 1 / (1 + np.exp(-o))                      
        kk = ((r[t] - y[t])* x[:,t])                                
        kk = kk[:,np.newaxis]
        #kk = (r[t] - y[t])* g
        del_w = del_w + kk
        
        
    #print('del_w\n',del_w)
    w = w + ita * del_w
    
    #if linalg.norm(del_w) < 0.01:
    #    break
 
    #print('w=', w)
    print('epoch=', epoch)

