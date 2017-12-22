# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:25:07 2017

@author: Jack
"""

import matplotlib.pyplot as plt
import functions as fun

# Read dataset
x, y = [], []
for sample in open("./data/prices.txt", "r"):
    xx, yy = sample.split(",")
    x.append(float(xx))
    y.append(float(yy))
x, y = fun.np.array(x), fun.np.array(y)

# Perform normalization
x = (x - x.mean()) / x.std()

# Scatter dataset
plt.figure()
plt.scatter(x, y, c="r", s=20)
plt.show()

# Set degrees
test_set = (1, 4, 10)

# show line's x
s_x = fun.np.linspace(-2, 4, 100)

# Visualize results
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    s_y = fun.get_model(d, x, y, s_x)()
    plt.plot(s_x, s_y, label="degree = {}".format(d))
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
plt.legend()
plt.show()

#cost
print('cost')
for d in test_set:
    print(fun.get_cost(d, x, y))