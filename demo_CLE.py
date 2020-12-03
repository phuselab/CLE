import numpy as np
from CLE import CLE
import skimage as sk
import matplotlib.pyplot as plt

#Data path
sal_path = 'img/deersalmap.jpg'
img_path = 'img/deer.jpg'

#Load image and relative saliency map
img = sk.io.imread(img_path)
sal = sk.io.imread(sal_path, as_gray=True)
sal = sk.transform.resize(sal, (img.shape[0], img.shape[1]))

print(img.shape)
print(sal.shape)

#Define CLE object
cle = CLE(saliecyMap=sal)
scan = cle.generateScanpath(sal=sal, numSteps=100)

#Plot generated scanpath on image
plt.subplot(1,2,1)
plt.imshow(img)
plt.imshow(sal, cmap='jet', alpha=0.5) # interpolation='none'
plt.subplot(1,2,2)
plt.imshow(img)
plt.plot(scan[:,1], scan[:,0], '-yo')
plt.show()

