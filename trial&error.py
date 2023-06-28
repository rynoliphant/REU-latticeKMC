import numpy as np
xx = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[4,5,6]])
x = np.array([[[1,2,3],[4,5,6],[7,8,9],[1,2,3],[10,11,12]]])
y = np.repeat(x, len(x[0]),axis=0)
trans_y = np.repeat(np.reshape(x,(len(x[0]),1,3)),len(x[0]),axis=1)

dist = np.linalg.norm(y-trans_y,axis=2)
nearest = [y[i][dist[i]==0] for i,r in enumerate(y)]
#other = np.concatenate(y[i][dist[i]==0])
print(np.where(xx==np.array([4,5,6]))[0][0])