import matplotlib.pyplot as plt

def membership_function(x):
    if x < 0:
        return 0
    elif x < 0.5:
        return 2 * x
    elif x < 1:
        return 2 - 2 * x
    else:
        return 0

x_values = [i / 100 for i in range(-100, 200)]
y_values = [membership_function(x) for x in x_values]

plt.plot(x_values, y_values)
plt.show()
