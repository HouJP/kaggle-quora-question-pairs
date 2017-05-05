import math
import numpy as np
from scipy.optimize import fsolve

def f(x):
    k, r1, r2 = x[0], x[1], x[2]
    lr_1 = 0.3
    rr_1 = 0.4
    score_1 = 0.47375
    lr_2 = 0.3
    rr_2 = 0.6
    score_2 = 0.43138
    lr_3 = 0.4
    rr_3 = 0.7
    score_3 = 0.52353
    return [
            k * (r1 * math.log(lr_1) + (1 - r1) * math.log(1 - lr_1)) + (1 - k) * (r2 * math.log(rr_1) + (1 - r2) * math.log(1 - rr_1)) + score_1,
            k * (r1 * math.log(lr_2) + (1 - r1) * math.log(1 - lr_2)) + (1 - k) * (r2 * math.log(rr_2) + (1 - r2) * math.log(1 - rr_2)) + score_2,
            k * (r1 * math.log(lr_3) + (1 - r1) * math.log(1 - lr_3)) + (1 - k) * (r2 * math.log(rr_3) + (1 - r2) * math.log(1 - rr_3)) + score_3
            ]

result = fsolve(f, [0.3, 0.12, 0.28])

print ('the result is',result)
print ('the error is',f(result))
