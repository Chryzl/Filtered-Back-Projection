import numpy as np
from math import cos, pi
import matplotlib.pyplot as plt

def ramp(dimension):
    return np.array(list(map(abs, np.linspace(-1, 1, dimension))))


def lak(dimension):
    def lak_helper(x):
        x = abs(x)
        if x < 0.5:
            return 1
        
        return 0
    return np.array(list(map(lak_helper, np.linspace(-1, 1, dimension))))

def ram_lak(dimension):
    return np.multiply(ramp(dimension), lak(dimension))

def cosine(dimension):
    cosine = np.array(list(map(cos, np.linspace(-pi/2, pi/2, dimension))))
    return np.multiply(cosine, ramp(dimension))

def hann(dimension):
    # the formula is taken form wikipedia, see: https://en.wikipedia.org/wiki/Hann_function
    def hann_helper(x):
        L = 1
        if abs(x) < L/2:
            return (1/L) * (cos((pi*x)/L))**2
        
        return 0
    
    hann = np.array(list(map(hann_helper, np.linspace(-1, 1, dimension))))
    return np.multiply(hann, ramp(dimension))
    

def plot_all():
    plt.plot(ramp(400), label="ramp")
    plt.plot(lak(400), label="lak")
    plt.plot(ram_lak(400), label="ram-lak")
    plt.plot(cosine(400), label="cosine")
    plt.plot(hann(400), label="hann")
    plt.legend()
    plt.show()