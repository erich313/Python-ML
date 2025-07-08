import pickle

f = open('frozen_lake8x8.pkl', 'rb')
q = pickle.load(f)
f.close()

num = 8

for i in range(num):
    for j in range(num):
        print(f'state{i*num+j}: {q[i*num+j]}')


