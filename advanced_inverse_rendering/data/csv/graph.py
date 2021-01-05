import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rotate_1axis.csv', names=['i', 'err', 'angle'])
plt.plot(range(300), df['err'])
plt.show()