import matplotlib.pyplot as plt 
import pandas as pd
import scipy

# Read DAT file
df = pd.read_csv('BasicDQN_Board2_ErrorLog.dat', sep='\s+', header=None, skiprows=0)

# Grab left-most column
ys = list(df.to_dict()[1].values())

# Create x-values 0-N for iteration count
xs = range(len(ys))

# Plot values
# plt.plot(xs, ys)
plt.plot(xs, ys, 'b')
plt.xlabel('Iterations')
plt.ylabel('Number of Errors')
plt.title('Deep Q-Learning')
plt.show()
