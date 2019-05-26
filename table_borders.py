from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image(im, h=8, title='', **kwargs):
   
    y = im.shape[0]
    x = im.shape[1]
    w = (y / x) * h
    plt.figure(figsize=(w, h))
    plt.imshow(im, interpolation="none", **kwargs)

    plt.axis('off')
    plt.title(title)
    plt.show()


image = cv2.imread('data/rgb.png')
print(type(image))
print(image.shape)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(np.all(rgb[..., 0] == image[..., 2]))
print(np.all(rgb[..., 1] == image[..., 1]))
print(np.all(rgb[..., 2] == image[..., 0]))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)
plot_image(gray, title='Gray')

_, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
_, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
_, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

_, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [gray, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


res = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
print(f'original image shape: {image.shape}')
print(f'changed image shape: {res.shape}')


kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(image, -1, kernel)

plt.subplot(121)
plt.imshow(image)
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(dst)
plt.title('Averaging')
plt.xticks([]), plt.yticks([])

plt.show()

blur = cv2.blur(image, (5, 5))

plt.subplot(121)
plt.imshow(image)
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(blur)
plt.title('Blurred')
plt.xticks([]), plt.yticks([])

plt.show()


blur = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

median = cv2.medianBlur(image, 5)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(gray, kernel, iterations=1)
plot_image(erosion, title='Erosion')

dilation = cv2.dilate(gray, kernel, iterations=1)
plot_image(dilation, title='Dilation')

opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
plot_image(opening, title='Opening')

closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
plot_image(closing, title='Closing')

edges = cv2.Canny(gray, 100, 200)
plot_image(edges, title='Detected Edges')


def contours_to_segments(contours: List[np.ndarray],
                         x_margin: int = 0,
                         y_margin: int = 0) -> List[Tuple]:

    segments = list(map(cv2.boundingRect, contours))
    return [
        (max(0, r[0] - x_margin), max(0, r[1] - y_margin), r[2] + x_margin, r[3] + y_margin)
        for r in segments]


def draw_segments(im: np.ndarray,
                  segments: List[Tuple],
                  color: Tuple = (255, 0, 0),
                  line_width: int = 3,
                  output_path: str = None):

    image = im.copy()
    for segment in segments:
        x, y, w, h = segment
        cv2.rectangle(image, (x, y), (x + w, y + h), color, line_width)

    plot_image(image)

    if output_path:
        cv2.imwrite(output_path, image)


image = cv2.imread('data/text_image.jpg')
plot_image(image, title='Text image')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, binary = cv2.threshold(np.invert(gray), 0, 255, 0 | 8)

plot_image(binary, title='Binary image')


kernel_h = np.ones((2, 4), np.uint8)
temp_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=2)


kernel_v = np.ones((1, 5), np.uint8)
line_img = cv2.dilate(temp_img, kernel_v, iterations=5)


_, contours, _ = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


segments = contours_to_segments(contours, x_margin=1, y_margin=5)


draw_segments(image, segments, output_path='data/segmented_text_image.png')
