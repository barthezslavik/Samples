import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('data/save/roi.csv')

# Create a line plot
plt.plot(df['period'], df['roi1'], label='roi1')
plt.plot(df['period'], df['roi2'], label='roi2')
plt.plot(df['period'], df['roi3'], label='roi3')
plt.plot(df['period'], df['roi'], label='roi')

# Add labels and title
plt.xlabel('Period')
plt.ylabel('Value')
plt.title('Line Plot')
plt.legend()

# Display the plot
plt.show()