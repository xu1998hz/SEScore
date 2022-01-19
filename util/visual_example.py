import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(60, 13, 500)

plt.hist(data, orientation = 'horizontal')
plt.grid()
plt.axis([0, 150, 0, 120])
plt.xticks(np.arange(0,151,25))

plt.show()
