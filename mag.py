import numpy as np
epsilon=1
for i in range(50):
    epsilon = max(epsilon * 0.995, 0.1)
    print(epsilon)
