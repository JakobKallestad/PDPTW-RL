"""
import numpy as np

dataset_size = 50
pdp_size = 20

locs = np.random.uniform(size=(dataset_size, pdp_size, 2)).tolist()  # Node locations

a = np.random.randint(1, 10, size=(dataset_size, pdp_size//2)).repeat(2, axis=1)
a[:, 1::2] *= -1  # Demand, uniform integer 1 ... 9
a = a.tolist()


type = np.tile([True, False], (pdp_size, pdp_size//2)).tolist()

p_or_d = np.empty((50, 20, 2))
p_or_d[:, ::2, :] = locs[:, 1::2, :]
p_or_d[:, 1::2, :] = locs[:, ::2, :]
p_or_d = p_or_d.tolist()
"""

"""
locs = [
    [0.0816, 0.7246],
    [4.0732e-01, 6.0775e-01],
    [3.8110e-01, 1.7268e-01],
    [5.6638e-01, 8.3864e-01],
    [1.9978e-01, 9.3829e-01],
    [6.2306e-01, 9.9302e-01],
    [4.1706e-02, 2.1394e-02],
    [6.6963e-01, 3.3555e-02],
    [9.0252e-02, 7.5424e-01],
    [7.4271e-01, 4.4390e-02],
    [8.5979e-01, 4.0278e-01],
    [8.7121e-01, 2.0658e-01],
    [2.4565e-02, 1.6854e-01],
    [4.2249e-01, 6.3777e-04],
    [4.6593e-01, 2.5046e-01],
    [3.8362e-01, 4.9993e-01],
    [5.2806e-01, 1.0909e-02],
    [7.5719e-01, 1.9355e-01],
    [8.0471e-01, 9.5816e-01],
    [3.3325e-01, 5.5612e-01],
    [7.7569e-01, 4.6039e-01]
]
"""


locs = [
    [0.7528, 0.8988],
    [0.7837, 0.5631],
    [0.7749, 0.8208],
    [0.2793, 0.6817],
    [0.2837, 0.6567],
    [0.2388, 0.7313],
    [0.6012, 0.3043],
    [0.2548, 0.6294],
    [0.9665, 0.7399],
    [0.4517, 0.4757],
    [0.7842, 0.1525],
    [0.6662, 0.3343],
    [0.7893, 0.3216],
    [0.5247, 0.6688],
    [0.8436, 0.4265],
    [0.9561, 0.0770],
    [0.4108, 0.0014],
    [0.5414, 0.6419],
    [0.2976, 0.7077],
    [0.4189, 0.0655],
    [0.8839, 0.8083]
]



locs = [[0.3144, 0.4608],
        [0.3094, 0.8841],
        [0.6314, 0.9475],
        [0.9515, 0.5817],
        [0.2920, 0.8040],
        [0.4826, 0.8701],
        [0.3932, 0.9422],
        [0.5660, 0.3889],
        [0.1738, 0.8700],
        [0.5650, 0.6730],
        [0.7021, 0.4742],
        [0.2486, 0.3508],
        [0.4777, 0.8105],
        [0.4635, 0.3732],
        [0.4320, 0.5821],
        [0.8991, 0.8773],
        [0.4708, 0.5663],
        [0.9534, 0.7197],
        [0.4472, 0.7964],
        [0.7277, 0.3657],
        [0.7023, 0.5396]]

locs = [
    [0.7528, 0.8988],
    [0.7837, 0.5631],
    [0.7749, 0.8208],
    [0.2793, 0.6817],
    [0.2837, 0.6567],
    [0.2388, 0.7313],
    [0.6012, 0.3043],
    [0.2548, 0.6294],
    [0.9665, 0.7399],
    [0.4517, 0.4757],
    [0.7842, 0.1525],
    [0.6662, 0.3343],
    [0.7893, 0.3216],
    [0.5247, 0.6688],
    [0.8436, 0.4265],
    [0.9561, 0.0770],
    [0.4108, 0.0014],
    [0.5414, 0.6419],
    [0.2976, 0.7077],
    [0.4189, 0.0655],
    [0.8839, 0.8083]
]

"""
locs = [
    [0.4416, 0.3634],
    [0.6604, 0.1303],
    [0.3498, 0.3824],
    [0.8043, 0.3186],
    [0.2908, 0.4196],
    [0.3728, 0.3769],
    [0.0108, 0.9455],
    [0.7661, 0.2634],
    [0.1880, 0.5174],
    [0.7849, 0.1412],
    [0.3112, 0.7091],
    [0.1775, 0.4443],
    [0.1230, 0.9638],
    [0.7695, 0.0378],
    [0.2239, 0.6772],
    [0.5274, 0.6325],
    [0.0910, 0.2323],
    [0.7269, 0.1187],
    [0.3951, 0.7199],
    [0.7595, 0.5311],
    [0.6449, 0.7224]]
"""


locs = [
    [0.7528, 0.8988],
    [0.7837, 0.5631],
    [0.7749, 0.8208],
    [0.2793, 0.6817],
    [0.2837, 0.6567],
    [0.2388, 0.7313],
    [0.6012, 0.3043],
    [0.2548, 0.6294],
    [0.9665, 0.7399],
    [0.4517, 0.4757],
    [0.7842, 0.1525],
    [0.6662, 0.3343],
    [0.7893, 0.3216],
    [0.5247, 0.6688],
    [0.8436, 0.4265],
    [0.9561, 0.0770],
    [0.4108, 0.0014],
    [0.5414, 0.6419],
    [0.2976, 0.7077],
    [0.4189, 0.0655],
    [0.8839, 0.8083]
]


locs = [[0.9957, 0.9365], [0.0349, 0.8517],
        [0.0107, 0.4809],
        [0.6907, 0.2713],
        [0.2306, 0.5714],
        [0.1388, 0.4355],
        [0.2112, 0.5611],
        [0.1919, 0.8377],
        [0.2552, 0.5858],
        [0.3467, 0.7633],
        [0.0994, 0.2824],
        [0.7437, 0.4059],
        [0.9790, 0.6924],
        [0.1928, 0.8466],
        [0.1052, 0.1923],
        [0.9196, 0.2765],
        [0.7743, 0.0106],
        [0.8660, 0.5071],
        [0.2743, 0.7691],
        [0.5722, 0.0760],
        [0.6641, 0.2621]]


#tour = [5, 15,  9, 17, 18, 19,  7, 10,  6,  1,  3, 16, 11, 12,  2, 20,  4, 13, 14,  8,  0]
#tour = [0, 1, 15, 11, 12, 16, 19,  9,  7,  3,  4,  5, 13, 17,  2, 20,  8, 14, 10, 6, 18,  0]
#tour = [0, 11, 13,  7, 19, 20,  9, 14, 12,  5,  6,  8,  1,  2, 15, 17,  3, 10, 16, 18,  4,  0]
#tour = [0, 17, 18,  5,  3,  4,  9, 13,  6, 14, 15, 10, 16, 19, 11, 12,  1, 20,  2, 7,  8,  0]
#tour = [0, 13,  9, 17,  7, 18, 15, 10, 14,  8,  5, 16, 11,  6, 12, 19, 20,  3,  1, 2,  4,  0]
#tour = [0, 17,  3,  5, 18,  4,  7, 13,  9,  6, 11, 12, 14, 15, 10, 16, 19,  1,  8, 20,  2,  0]
#tour = [0, 17, 18,  3,  5,  7,  4, 13,  9,  6, 11, 12, 14, 15, 10, 16, 19,  1,  8, 20,  2,  0]
#tour = [0, 17,  3,  5, 18,  4,  7, 13,  9,  6, 11, 12, 14, 15, 10, 16, 19,  1,  8, 20,  2,  0]  #4.0298 (RL), 4.0295810782375066 (Here)
#tour = [0, 17,  3,  5, 18,  4,  7, 13,  9,  6, 11, 12, 14, 15, 10, 16, 19,  1,  8, 20,  2,  0]   #4.0298 (RL without last line),
#tour = [0, 17,  3,  5, 18,  4,  9, 13,  6, 11, 14, 12, 15, 10, 16, 19,  7,  1,  8, 20,  2,  0]    #4.6177 (RL without last line),  4.617429410825182 (Here)
#tour = [0, 17,  3,  4,  5, 18,  7, 13,  9,  6, 11, 14, 12, 15, 10, 16, 19,  1,  8, 20,  2,  0]     #4.0800 (RL),    4.079694483841318 (Here)
#tour = [0, 17,  3, 18,  5,  7,  4, 13,  9,  6, 11, 12, 14, 15, 10, 16, 19,  1,  8, 20,  2,  0]    #4.0165 (RL),4.016172246427002 (Here)

tour = [0, 17,  3,  4, 18, 13,  7,  1,  2, 14,  5,  8,  6,  9, 10, 19, 20, 15, 16, 11, 12,  0]    #6.0132 (RL), 6.012962196855224 (Here)
tour = [0, 17,  3,  4, 18, 13,  9,  1,  2, 10, 14,  5,  6,  7,  8, 15, 16, 19, 20, 11, 12,  0]    #5.9485 (RL), 5.948127257499651 (Here)
tour = [0, 17,  3,  4, 18,  7, 13,  1,  2, 14,  5,  6,  8,  9, 10, 19, 20, 15, 16, 11, 12,  0]    #5.9195 (RL), 5.91923708066929 (Here)

# THEN I modified get_cost() function. So now hopefully the RL results should match that of this algo with "len(tour)-2":

tour = [0, 17,  3,  4, 18,  7, 13,  1,  2, 14,  5,  6,  8,  9, 10, 19, 20, 15, 16, 11, 12,  0]    #5.6747 (RL),  5.67456648578711 (Here)
tour = [0, 17,  3,  4, 18,  9, 13,  1,  2, 10, 14,  5,  6,  7,  8, 15, 16, 19, 20, 11, 12,  0]    #5.4978 (RL),  5.497516216758673 (Here)



total_dist = 0
for i in range(len(tour)-2):
    origin = tour[i]
    dest = tour[i+1]
    dist = ((locs[origin][0] - locs[dest][0])**2 + (locs[origin][1] - locs[dest][1])**2 )**0.5
    total_dist += dist
print(total_dist)

