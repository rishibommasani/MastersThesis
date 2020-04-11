import matplotlib.pyplot as plt
import numpy as np

x = [5, 50, 100, 150, 200, 250, 300]
standard = [50.8, 50.6, 52.5, 52.4, 54.4, 53.3, 50.7]
random = [52.7, 53.6, 52.4, 54.4, 53.4, 56.3, 52.7]
standard1 = [60.2]*7
random1 = [61.2]*7

plt.plot(x, standard, label = 'RG65 System')
plt.plot(x, random, label = 'WS353 System')
plt.plot(x, standard1, label = 'RG65 Original')
plt.plot(x, random1, label = 'WS353 Original')

plt.legend()
plt.suptitle('Manifold Dimension Experiment')
plt.xlabel('Manifold Dimension')
plt.ylabel('Spearman Rank Correlation')
plt.show()