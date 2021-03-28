from generate_pdp_data import load_dataset
import torch
import math

dataset = load_dataset('../data/pdp/pdp20_TEST1_seed1234.pkl')

data = dataset[0]

#print(data)

loc = data['loc']

print(loc)

dists = (loc[1::2] - loc[0:-1:2]).norm(p=2, dim=1)
dummies = torch.zeros(10)

print(dists)
print(dummies)

merged = torch.flatten(torch.stack((dists, dummies), dim=1), start_dim=0, end_dim=1)
print(merged)


a = [0.7837, 0.5631] # -1.5366613565751117 -88.04421027260157    1.6049312970146814 91.95578972739843
b = [0.7749, 0.8208]
c = [0.2793, 0.6817] # 1.7450121132378411 99.98182928773319      4.8866047668276344 279.9818292877332
d = [0.2837, 0.6567]
e = [0.2388, 0.7313] # 2.2745421325785906 130.32166452143898     5.416134786168383 310.32166452143895
f = [0.6012, 0.3043]

aa = loc[0]
bb = loc[1]
cc = loc[2]
dd = loc[3]
ee = loc[4]
ff = loc[5]

print(a, aa)
print(b, bb)
print(c, cc)
print(d, dd)
print(e, ee)
print(f, ff)


print("debug: ", a[1] - b[1], a[0]-b[0])

myradians = math.atan2(ee[1]-ff[1], ee[0]-ff[0]) + math.pi
mydegrees = math.degrees(myradians)
print(myradians, mydegrees)

result = aa - bb
result2 = aa - bb
result3 = aa - bb

#print(torch.split(aa-bb, 1)[::-1])

rads2 = torch.atan2(*torch.split(aa-bb, 1))

#rads2 = torch.atan2(torch.tensor([-0.25769999999999993], dtype=torch.float32), torch.tensor([0.008799999999999919], dtype=torch.float32))
#convert = (180/math.pi)
print(rads2, (1.5*math.pi - rads2))

diffs = (loc[0:-1:2] - loc[1::2])
rads3 = (1.5*math.pi - torch.atan2(diffs[:, 0], diffs[:, 1]))
print(rads3)


a = (1, 4, 5, 6, 7)


