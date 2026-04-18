import numpy as np
data = np.load('travel_time_matrix.npy')
#print(data)

from matplotlib import pyplot as plt
plt.imshow(data, cmap='gray')
plt.show()
