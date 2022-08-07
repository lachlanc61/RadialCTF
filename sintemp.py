import matplotlib.pyplot as plt
import numpy as np
wavelength = 0.5
x = np.linspace(-1, 1, 1000)
y = abs(np.sin(2 * np.pi * x / wavelength))
plt.plot(x, y)
plt.show()