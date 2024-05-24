import sys
import random

s = sys.argv[1]

def U(a, b):
    # Uniform, lower bound a, upper bound b
    out = random.uniform(a, b)
    if isinstance(a, int) and isinstance(b, int):
        return round(out)
    else:
        return out
    
def N(a, b):
    # Normal, mean a, standard deviation b
    out = random.normalvariate(a, b)
    if isinstance(a, int) and isinstance(b, int):
        return round(out)
    else:
        return out

print(eval(s))
