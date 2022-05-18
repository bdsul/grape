#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:31:49 2021

@author: allan
"""

import numpy as np
import math

def WA(a, b, x):
    return x*a+(1-x)*b

def OWA(a, b, x):
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

def minimum(a, b):
    return np.minimum(a, b)

def maximum(a, b):
    return np.maximum(a, b)
    
def dilator(b):
    return b**0.5

def concentrator(b):
    return b**2

def div(a, b):
    result = np.where(a == 0, np.ones_like(a), a / b)
    if result == float("inf"):
        raise ZeroDivisionError
    else:
        return result
    
def exp(a):
    result = np.exp(a)
    if result == float("inf"):
        raise OverflowError
    else:
        return result

def neg(a):
    return -a

def pdiv(a, b):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(a == 0, np.ones_like(a), a / b)
    except ZeroDivisionError:
        # In this case we are trying to divide two constants, one of which is 0
        # Return a constant.
        return 1.0
    
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
def psin(n):
    return np.sin(n)

def pcos(n):
    return np.cos(n)

def add(a, b):
    return np.add(a,b)

def sub(a, b):
    return np.subtract(a,b)

def mul(a, b):
    return np.multiply(a,b)

def psqrt(a):
    return np.sqrt(abs(a))

def max_(a,b):
    return np.maximum(a, b)

def min_(a,b):
    return np.minimum(a, b)

def plog(a):
    return np.log(1.0 + np.abs(a))

def not_(a):
    return np.logical_not(a)

def and_(a, b):
    return np.logical_and(a,b)

def or_(a, b):
    return np.logical_or(a,b)

def nand_(a, b):
    return np.logical_not(np.logical_and(a,b))

def nor_(a, b):
    return np.logical_not(np.logical_or(a,b))

def greater_than_or_equal(a, b):
    return a >= b

def less_than_or_equal(a, b):
    return a <= b

def if_(i, o0, o1):
    """If _ than _ else _"""
    return np.where(i, o0, o1)