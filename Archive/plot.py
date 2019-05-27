import matplotlib.pyplot as plt 
import pandas as pd
import scipy
import re

filename = 'DeepQLearning_map2_stats.dat'

# Read DAT file
df = pd.read_csv(filename, sep='\s+', header=None, skiprows=0)

# Grab left-most column
ys = list(df.to_dict()[1].values())

# Create x-values 0-N for iteration count
xs = range(len(ys))

# Plot values
plt.plot(xs, ys, 'b')
plt.xlabel('Iterations')
plt.ylabel('Number of Errors')
plt.title('Deep Q-Learning')

# Plot is saved as PNG with same name as DAT file
plt.savefig(re.sub(r'\.dat$', '.png', filename))
plt.show()
