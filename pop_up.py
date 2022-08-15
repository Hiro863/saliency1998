import cv2
import numpy as np
from matplotlib import pyplot as plt
import random



def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def draw_rectangle(image, color, orientation):

    width = int(image.shape[0] / 4)
    length = int(image.shape[0] * 0.8)
    gap1 = int(image.shape[0] * 3 / 8)
    gap2 = int(image.shape[0] * 0.1)

    cv2.rectangle(image, (gap1, gap2), (gap1 + width, gap2 + length), color, -1)

    return rotate_image(image, orientation)

size = (5000, 5000, 3)
orientation_image = np.zeros(size).astype(np.uint8)
color_image = np.zeros(size).astype(np.uint8)
conjunctive_image = np.zeros(size).astype(np.uint8)

red = (0, 0, 255)
green = (0, 255, 0)

# orientation pop up
num_image = 10
odd_one_out = random.randint(0, num_image * num_image - 1)
grid = int(size[0] / num_image)
for i in range(num_image):
    for j in range(num_image):

        if 10 * i + j == odd_one_out:
            # odd one out
            random_orientation = random.randint(70, 90)
        else:
            # normal orientation
            random_orientation = random.randint(35, 40)

        small = np.zeros((int(size[0]/10), int(size[1]/10), 3))
        small = draw_rectangle(small, green, random_orientation)

        orientation_image[i * grid: (i + 1) * grid, j * grid: (j + 1) * grid, :] = small

# color pop up
num_image = 10
odd_one_out = random.randint(0, num_image * num_image - 1)
grid = int(size[0] / num_image)
for i in range(num_image):
    for j in range(num_image):

        if 10 * i + j == odd_one_out:
            # odd one out
            color = red
        else:
            # normal color
            color = green
        random_orientation = random.randint(0, 90)
        small = np.zeros((int(size[0]/10), int(size[1]/10), 3))
        small = draw_rectangle(small, color, random_orientation)

        color_image[i * grid: (i + 1) * grid, j * grid: (j + 1) * grid, :] = small


# conjunctive pop up
num_image = 10
odd_one_out = random.randint(0, num_image * num_image - 1)
grid = int(size[0] / num_image)
for i in range(num_image):
    for j in range(num_image):

        if 10 * i + j == odd_one_out:
            # odd one out
            color = red
        else:
            # normal color
            color = green
        random_orientation = random.randint(0, 90)
        small = np.zeros((int(size[0]/10), int(size[1]/10), 3))
        small = draw_rectangle(small, color, random_orientation)

        color_image[i * grid: (i + 1) * grid, j * grid: (j + 1) * grid, :] = small

plt.imshow(color_image)
plt.show()

cv2.imwrite('orientation_pop_up.jpg', orientation_image)
cv2.imwrite('color_pop_up.jpg', color_image)