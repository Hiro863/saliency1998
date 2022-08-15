from visual_attention2000 import Saliency
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

#random.seed(0)

# colors
red = (0, 0, 255)
green = (0, 255, 0)

# number of bars par row
num_image = 10

# image size
size = (5000, 5000, 3)


def rotate_image(image, angle):
    # rotates the rectangle at an angle
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def draw_rectangle(image, color, orientation):
    # width of the bar
    width = int(image.shape[0] / 4)

    # length of the bar
    length = int(image.shape[0] * 0.8)

    # gap between the bar and the bounding square horizontally
    gap1 = int(image.shape[0] * 3 / 8)

    # gap between the bar and the bounding square vertically
    gap2 = int(image.shape[0] * 0.1)

    # draw rectangle
    cv2.rectangle(image, (gap1, gap2), (gap1 + width, gap2 + length), color, -1)

    return rotate_image(image, orientation)


def orientation_pop_up(normal, exception):
    # draws a orientation pop up image
    # normal: range of orientations for normal bars, e.g. 45 degrees to 90 degrees
    # exception: range of orientations for odd one out

    # the image
    orientation_image = np.zeros(size).astype(np.uint8)

    # index of the odd one out, counting from the top left
    odd_one_out = random.randint(0, num_image * num_image - 1)

    # grid size, where each bar is found
    grid = int(size[0] / num_image)

    for i in range(num_image):
        for j in range(num_image):

            if 10 * i + j == odd_one_out:
                # odd one out
                random_orientation = random.randint(exception[0], exception[1])
            else:
                # normal orientation
                random_orientation = random.randint(normal[0], normal[1])

            # small image with a single bar
            small = np.zeros((grid, grid, 3))
            small = draw_rectangle(small, green, random_orientation)

            orientation_image[i * grid: (i + 1) * grid, j * grid: (j + 1) * grid, :] = small

    return orientation_image, odd_one_out


def color_pop_up():
    # draws a orientation pop up image

    # the image
    color_image = np.zeros(size).astype(np.uint8)

    # index of the odd one out, counting from top left
    odd_one_out = random.randint(0, num_image * num_image - 1)

    # grid size, where each bar is found
    grid = int(size[0] / num_image)

    for i in range(num_image):
        for j in range(num_image):

            if 10 * i + j == odd_one_out:
                # odd one out
                color = red
            else:
                # normal color
                color = green

            # the orientation is toatally random
            random_orientation = random.randint(0, 90)

            # small image with a single bar
            small = np.zeros((int(size[0]/10), int(size[1]/10), 3))
            small = draw_rectangle(small, color, random_orientation)

            color_image[i * grid: (i + 1) * grid, j * grid: (j + 1) * grid, :] = small

    return color_image, odd_one_out


def check_correct(correct, answer):
    # check if the answer is correct
    # correct is index
    # answer is a coordinate

    # find square location
    row = correct // 10
    col = correct % 10

    # correct region
    grid = int(size[0] / num_image / (2**4))
    correct_row_min = row * grid
    correct_row_max = (row + 1) * grid
    correct_col_min = col * grid
    correct_col_max = (col + 1) * grid

    if correct_row_min <= answer[0] <= correct_row_max and correct_col_min <= answer[1] <= correct_col_max:
        return True
    else:
        return False


if __name__ == '__main__':

    # trial parameters
    trial_type = 'orientation'
    accuracy = 0
    num_trials = 20

    for _ in range(num_trials):

        # load orientation image
        if trial_type == 'orientation':
            image, correct = orientation_pop_up((40, 50), (90, 100))
        if trial_type == 'color':
            image, correct = color_pop_up()

        # declare  the Saliency object
        foa_finder = Saliency(image.shape)

        # update
        foa_finder.update(image, inhibit=False)

        # find foa coordinate
        coordinate = np.unravel_index(np.argmax(foa_finder.saliency_map, axis=None), foa_finder.saliency_map.shape)

        # check answer
        result = check_correct(correct, coordinate)
        print(result)
        accuracy += result

        # print saliency map
        plt.imshow(foa_finder.saliency_map)
        plt.show()

    # print accuracy
    accuracy = accuracy / num_trials
    print(accuracy)
    cv2.destroyAllWindows()

