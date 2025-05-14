'''
import numpy as np

custom_map=[
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
]

# 0 1 2 3 4 5 6 7 -0
# 8 9 10 11 12 13 14 15 -1
# 16 17 18 19 20 21 22 23 -2
# 24 25 26 27 28 29 30 31 -3
# 32 33 34 35 36 37 38 39 -4
# 40 41 42 43 44 45 46 47 -5
# 48 49 50 51 52 53 54 55 -6
# 56 57 58 59 60 61 62 63 -7
l = []
for control in range(4, -1, -1):
    for i in range(control, 8):
        for j in range(control, 8):
            if custom_map[i][j] == "F" or custom_map[i][j] == "F":
                l.append(8*i+j)
    print(l)
    break
    # l = []

print(np.random.choice(l))
'''

