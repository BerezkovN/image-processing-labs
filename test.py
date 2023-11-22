import numpy as np


rows = 10 
cols = 10
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
r = 4 # Radius of the mask
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 < r*r
mask[mask_area] = 0

print(mask)