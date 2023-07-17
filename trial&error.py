import numpy as np
class idk:
    def __init__ (self, name):
        self.name = name

class hello:
    def __init__ (self, num, num_2, num_3):
        self.num = [num, num_2, num_3]

xx = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[4,5,6]])
f = np.array([[[1,2,3],[4,5,6],[7,2,9],[1,2,3],[10,11,12]],
              [[1,0,3],[7,5,6],[7,8,9],[1,2,3],[10,11,12]]])

print(np.unique(xx))

#return = np.array([[[1,2,3], [4,5,6], [1,2,3], [10,11,12]],
#                   [[7,8,9], [1,2,3], [10,11,12]]])

#y = np.repeat(x, len(x[0]),axis=0)
#trans_y = np.repeat(np.reshape(x,(len(x[0]),1,3)),len(x[0]),axis=1)

#dist = np.linalg.norm(y-trans_y,axis=2)
#nearest = [y[i][dist[i]==0] for i,r in enumerate(y)]
#other = np.concatenate(y[i][dist[i]==0])

hi = idk('X')

new = hello(idk('Ga'),idk('N'),idk('Si'))
for value in new.num:
    if value.name == 'Ga':
        value = hi

a = np.array([[1,6,3],[6,8,9]])
b = np.array([[6,3,5],[2,35,8]])

output = np.concatenate((a,b), axis=1)

print(output)





new_xx = np.tile(xx,(len(f),1))

n_xx = np.array([",".join(item) for item in new_xx.astype(str)])
n_f = np.array([",".join(item) for item in f.reshape(10,3).astype(str)])
print(n_xx)
print(len(n_f))

sorter = n_xx.argsort(kind='mergesort')
i = sorter[np.searchsorted(n_xx, n_f,sorter=sorter)]
print(i-1)

#sorter = n_xx.argsort()
#print(sorter)
#rint(n_xx[sorter])
#i = sorter[np.searchsorted(n_xx,n_x, side='left', sorter=sorter)]
#rint(new_xx[i])
#i_newshape = i.reshape(2,5,3)
#rint(i_newshape)
#_bool = (i_newshape[:,:,2]-i_newshape[:,:,1])==1
#i_bool1 = (i_newshape[:,:,1]-i_newshape[:,:,0])==1
#otal_bool = i_bool*i_bool1
#new_i = (((i_newshape[total_bool][:,2]+1)/3)-1).astype(int)

#print(new_i)
rofae = np.array([[1,2,3,4,5,6],[6,5,4,3,2,1],[8,5,2,7,9,4]])
ello = np.array([[3,0,0],[0,1,0],[0,0,2]])
ello_new = np.tile(ello, (len(rofae),1,1))

uhhh = rofae[:,:3]-rofae[:,3:]

solve = np.linalg.solve(ello_new,uhhh)
print(uhhh)
print(solve)

print(np.min(rofae,axis=1, initial=10, where=(rofae!=2)))
print(np.reshape(np.repeat(np.min(rofae,axis=1, initial=10, where=(rofae!=2)),6), (3,6)))
