import numpy as np
class idk:
    def __init__ (self, name):
        self.name = name

class hello:
    def __init__ (self, num, num_2, num_3):
        self.num = [num, num_2, num_3]

xx = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[4,5,6]])
x = np.array([[[1,2,3],[4,5,6],[7,8,9],[1,2,3],[10,11,12]]])
y = np.repeat(x, len(x[0]),axis=0)
trans_y = np.repeat(np.reshape(x,(len(x[0]),1,3)),len(x[0]),axis=1)

dist = np.linalg.norm(y-trans_y,axis=2)
nearest = [y[i][dist[i]==0] for i,r in enumerate(y)]
#other = np.concatenate(y[i][dist[i]==0])

def whatever(um:hello):
    first = um.num[0]
    um.num[0] = um.num[1]
    um.num[1] = first
    return um
#print(whatever(xx)==xx)

new = hello(idk('Ga'),idk('N'),idk('Si'))
thing = new.num
print(thing)
new_new = whatever(new)
some = new_new.num
print(some)
print(thing)
print(new.num)