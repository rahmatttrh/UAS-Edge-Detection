# UAS 
# Rahmat Hidayat
# 171011400893



# Edge Detection

import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import roberts, sobel, scharr, prewitt

img = io.imread("images/myimage.jpg", as_gray=True)  
print(img.shape)


edge_roberts = roberts(img)

edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)


fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax[1].set_title('Rahmat Edge Detection')

ax[2].imshow(edge_sobel, cmap=plt.cm.gray)
ax[2].set_title('Rhd')

ax[3].imshow(edge_scharr, cmap=plt.cm.gray)
ax[3].set_title('Abf')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


from skimage import feature
edge_canny = feature.canny(img, sigma=3)
plt.imshow(edge_canny)