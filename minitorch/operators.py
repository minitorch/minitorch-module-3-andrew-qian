"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float: 
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -1.0 * x

def lt(x: float, y: float) -> float:
    return x < y

def eq(x: float, y: float) -> float:
    return x == y

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(neg(x))) if x >= 0 else exp(x) / (1.0 + exp(x))

def relu(x: float) -> float:
    return x if x > 0.0 else 0.0

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1.0 / x

def log_back(x : float, y: float) -> float:
    return inv(x) * y

def inv_back(x: float, y: float) -> float:
    return -1.0 * inv(x ** 2.0) * y

def relu_back(x: float, y: float) -> float:
    return y if x > 0.0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def apply(arr: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in arr:
            ret.append(fn(x))

        return ret
    
    return apply

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def apply(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(arr1, arr2):
            ret.append(fn(x, y))
        return ret
    
    return apply

def reduce(fn: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    def apply(arr: Iterable[float]) -> float:
        it = iter(arr)
        try:
            prev = next(it)
        except StopIteration:
            return 0.0

        res = prev
        for curr in it:
            res = fn(res, curr)

        return res
            
    return apply

def negList(arr: Iterable[float]) -> Iterable[float]:
    return map(neg)(arr)

def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(arr1, arr2)
    
def sum(arr: Iterable[float]) -> float:
    return reduce(add)(arr)

def prod(arr: Iterable[float]) -> float:
    return reduce(mul)(arr)