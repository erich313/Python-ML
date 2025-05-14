import pickle

f = open('frozen_lake8x8.pkl', 'rb')
q = pickle.load(f)
f.close()

for i in range(8):
    for j in range(8):
        print(f'state{i*8+j}: {q[i*8+j]}')


# print(q[0][0] == 0)