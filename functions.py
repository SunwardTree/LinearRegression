# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:25:07 2017

@author: Jack
"""

import numpy as np

# Get regression model under LSE criterion with degree 'deg'
# return show_y
def get_model(deg, x, y, show_x):
    return lambda : np.polyval(np.polyfit(x, y, deg), show_x)


# Get the cost of regression model above under given x, y
def get_cost(deg, x, y):
    show_x = x
    show_y = get_model(deg, x, y, show_x)()
    return 0.5 * ((show_y - y) ** 2).sum()
