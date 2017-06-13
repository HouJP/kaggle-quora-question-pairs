import math
import numpy as np
from scipy.optimize import fsolve


def f2(x):
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


def f3(x):
    k1, k2, r1, r2, r3 = x[0], x[1], x[2], x[3], x[4]

    # online clique_size <3 / =3 / >3

    s1_1 = 0.3
    s2_1 = 0.4
    s3_1 = 0.65
    score_1 = 0.42161

    s1_2 = 0.35
    s2_2 = 0.56
    s3_2 = 0.75
    score_2 = 0.46212

    s1_3 = 0.21
    s2_3 = 0.3
    s3_3 = 0.85
    score_3 = 0.31703

    s1_4 = 0.14
    s2_4 = 0.25
    s3_4 = 0.7
    score_4 = 0.28731

    s1_5 = 0.46
    s2_5 = 0.67
    s3_5 = 0.8
    score_5 = 0.59638

    s1_1 = 0.29
    s2_1 = 0.39
    s3_1 = 0.64
    score_1 = 0.48881

    s1_2 = 0.34
    s2_2 = 0.55
    s3_2 = 0.74
    score_2 = 0.64419

    s1_3 = 0.20
    s2_3 = 0.29
    s3_3 = 0.84
    score_3 = 0.38840

    s1_4 = 0.13
    s2_4 = 0.24
    s3_4 = 0.69
    score_4 = 0.34220

    s1_5 = 0.45
    s2_5 = 0.66
    s3_5 = 0.79
    score_5 = 0.82692

    return [
        k1 * (r1 * math.log(s1_1) + (1 - r1) * math.log(1 - s1_1)) + k2 * (r2 * math.log(s2_1) + (1 - r2) * math.log(1 - s2_1)) + (1 - k1 - k2) * (r3 * math.log(s3_1) + (1 - r3) * math.log(1 - s3_1)) + score_1,
        k1 * (r1 * math.log(s1_2) + (1 - r1) * math.log(1 - s1_2)) + k2 * (r2 * math.log(s2_2) + (1 - r2) * math.log(1 - s2_2)) + (1 - k1 - k2) * (r3 * math.log(s3_2) + (1 - r3) * math.log(1 - s3_2)) + score_2,
        k1 * (r1 * math.log(s1_3) + (1 - r1) * math.log(1 - s1_3)) + k2 * (r2 * math.log(s2_3) + (1 - r2) * math.log(1 - s2_3)) + (1 - k1 - k2) * (r3 * math.log(s3_3) + (1 - r3) * math.log(1 - s3_3)) + score_3,
        k1 * (r1 * math.log(s1_4) + (1 - r1) * math.log(1 - s1_4)) + k2 * (r2 * math.log(s2_4) + (1 - r2) * math.log(1 - s2_4)) + (1 - k1 - k2) * (r3 * math.log(s3_4) + (1 - r3) * math.log(1 - s3_4)) + score_4,
        k1 * (r1 * math.log(s1_5) + (1 - r1) * math.log(1 - s1_5)) + k2 * (r2 * math.log(s2_5) + (1 - r2) * math.log(1 - s2_5)) + (1 - k1 - k2) * (r3 * math.log(s3_5) + (1 - r3) * math.log(1 - s3_5)) + score_5
    ]


result2 = fsolve(f2, np.array([0.3, 0.12, 0.28]))

print ('the result is',result2)
print ('the error is',f2(result2))


result3 = fsolve(f3, np.array([0.3, 0.12, 0.28, 0.45, 0.67]))

print ('the result is', result3)
print ('the error is', f3(result3))

