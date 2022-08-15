import cv2
import numpy as np
import math
from scipy.ndimage import maximum_filter

class Saliency:
    def __init__(self, size, levels=9):
        # number of levels
        self.levels = levels

        # map size is median size of the image pyramid
        map_size = int(size[0] / 2 ** ((levels - 1) / 2)), int(size[1] / 2 ** ((levels - 1) / 2))

        # inhibition radius
        self.inhibition_radius = int(map_size[0] / 10)

        # circle radius
        self.radius = int(size[0] / 10)

        # WTA-threshold
        # TODO needs a more sophisticated way to define the threshold
        self.wta_threshold = 1000

        # saliency map
        self.saliency_map = np.zeros(map_size)

        # current saliency map
        self.wta_map = np.zeros(map_size)

        # current focus of attention
        self.foa = (0, 0)

        # remember last FOA for drawing purposes
        self.last_foa = (0, 0)

    def get_features(self, image, channel, levels=9):

        # image: original image
        # channel: a function which generates intensity, colour or orientation maps

        # original size
        start_size = image.shape

        # from the raw image obtain perticular channel
        image = channel(image)

        # image pyramid using Opencv
        num_levels = 9
        scales = [image]
        for level in range(num_levels - 1):
            scales.append(cv2.pyrDown(scales[-1]))

        features = []
        for c in range(2, num_levels - 4):
            # centre image, c taken from {2, 3, 4}
            centre_img = scales[c]

            for delta in (3, 4):
                # coarse surround-images, delta taken from {3, 4}
                surround_img = scales[c + delta]

                # image sizes
                src_size = surround_img.shape[1], surround_img.shape[0]
                dst_size = centre_img.shape[1], centre_img.shape[0]

                # rescale the surround-image to the original size
                surround_img = cv2.resize(src=surround_img, dsize=dst_size)

                # difference between the centre image and surround image
                features.append(cv2.absdiff(centre_img, surround_img))

        return features

    def dog_normalization(self, image, iteration=10):
        def dog(x, y, size):
            # relative strength
            c_ex = 0.5
            c_inh = 1.5

            # size of the bell curve
            sigma_ex = 0.02 * size[0]
            sigma_inh = 0.25 * size[0]

            # single Gaussian term
            def term(x, y, c, sigma):
                return c ** 2 * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)

            return term(x, y, c_ex, sigma_ex) - term(x, y, c_inh, sigma_inh)

        def dog_filter(image):
            return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

        if image.max() != 0:
            image = cv2.convertScaleAbs(image, alpha=1. / image.max(), beta=0.)

        kernel_size = int(0.25 * image.shape[0]), int(0.25 * image.shape[0])

        half_width = int(kernel_size[0] / 2)
        half_height = int(kernel_size[1] / 2)

        # kernel of the filter
        kernel = np.array([[dog(half_width - i, half_height - j, image.shape) for j in range(kernel_size[1])] for i in
                           range(kernel_size[1])])

        # main iteration
        for _ in range(iteration):
            image_convolved = dog_filter(image)
            image = image + image_convolved - 0.02
            cv2.threshold(src=image, dst=image, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)

        return image

    def sum_normalised_features(self, features, size, levels=9):

        # get the common size (median size)
        common_height = int(size[0] / 2 ** ((levels - 1) / 2))
        common_width = int(size[1] / 2 ** ((levels - 1) / 2))
        common_size = common_width, common_height

        # apply normalisation to the first feature
        conspicuity = self.dog_normalization(cv2.resize(features[0], common_size))

        # summation
        for feature in features[1:]:
            # normalise the next feature
            resized = self.dog_normalization(cv2.resize(feature, common_size))

            # increment the conspicuity map
            conspicuity = cv2.add(conspicuity, resized)
        return conspicuity

    def get_intensity(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def get_intensity_conspicuity(self, image):

        # get the features
        feature_list = self.get_features(image, channel=self.get_intensity)

        # sum over all six features
        return self.sum_normalised_features(feature_list, image.shape)

    def get_normalised_color_channels(self, image):
        # split the image
        red, green, blue = cv2.split(image)

        # hue are differentiated only if the intensity is above 1/10 of maximum intensity
        # otherwise it is set to zero
        threshold_ratio = 10.
        intens = self.get_intensity(image)
        threshold = intens.max() / threshold_ratio

        cv2.threshold(src=red, dst=red, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=green, dst=green, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=blue, dst=blue, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)

        # R = r - (g + b)/2
        tuned_red = red - (green + blue) / 2

        # G = g - (r + b)/2
        tuned_green = green - (red + blue) / 2

        # B = b - (r + g)/2
        tuned_blue = blue - (red + green) / 2

        # Y = (r + g)/2 - |r - g|/2 - b
        diff = cv2.absdiff(red, green)
        tuned_yellow = (red + green) / 2 - cv2.absdiff(red, green) / 2 - blue

        # negative values are set to zero
        cv2.threshold(src=tuned_red, dst=tuned_red, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=tuned_green, dst=tuned_green, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=tuned_blue, dst=tuned_blue, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=tuned_yellow, dst=tuned_yellow, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)

        # new RGBY image
        image = cv2.merge((tuned_red, tuned_green, tuned_blue, tuned_yellow))

        return image

    def rg_conspicuity(self, image):

        def difference_red_green(image):
            red, green, _, _ = cv2.split(image)
            return cv2.absdiff(red, green)

        feature_list = self.get_features(image=image, channel=difference_red_green)
        return self.sum_normalised_features(feature_list, image.shape)

    def by_conspicuity(self, image):

        def difference_blue_yellow(image):
            _, _, blue, yellow = cv2.split(image)
            return cv2.absdiff(blue, yellow)

        feature_list = self.get_features(image=image, channel=difference_blue_yellow)
        return self.sum_normalised_features(feature_list, image.shape)

    def get_color_conspicuity(self, image):

        # first convert from RGB to RGBY
        image = self.get_normalised_color_channels(image)

        # get RG and BY and return the sum
        rg = self.rg_conspicuity(image)
        by = self.by_conspicuity(image)
        return rg + by

    def get_gabor_filter(self, dims, lambd, theta, psi, sigma, gamma):

        def x_dash(x, y):
            return x * math.cos(theta) + y * math.sin(theta)

        def y_dash(x, y):
            return -x * math.sin(theta) + y * math.cos(theta)

        def gabor(x, y):
            x_ = x_dash(x, y)
            y_ = y_dash(x, y)
            return math.exp(-(x_ ** 2 + gamma ** 2 * y_ ** 2) / 2 * sigma ** 2) * math.cos(
                2 * math.pi * x_ / lambd + psi)

        half_width = dims[0] / 2
        half_height = dims[1] / 2

        # kernel of the filter
        kernel = np.array([[gabor(half_width - i, half_height - j) for j in range(dims[1])] for i in range(dims[1])])

        def gabor_filter(image):
            return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

        return gabor_filter, kernel

    # find conspicuity map for a single theta
    def gabor_conspicuity_single(self, image, theta):
        # the filter function
        gabor_filter, _ = self.get_gabor_filter(dims=(40, 40),
                                                lambd=20,
                                                theta=theta,
                                                psi=0,
                                                sigma=0.1,
                                                gamma=0.5)

        # features
        feature_list = self.get_features(image=self.get_intensity(image), channel=gabor_filter)
        return self.sum_normalised_features(feature_list, image.shape)

    def get_orientation_conspicuity(self, image):

        # orientation: 0, normalised before summation
        theta = 0
        conspicuity = self.dog_normalization(self.gabor_conspicuity_single(image, theta), iteration=30)

        # orientation: 45°, 90°, 135°
        # sum over all four orientations, normalised before summation
        for i in range(1, 4):
            theta += math.pi / 4
            conspicuity += self.dog_normalization(self.gabor_conspicuity_single(image, theta), iteration=30)

        return conspicuity

    def get_saliency(self, image):
        # from the input image creates a single saliency map

        # get three conspicuity maps
        intensity_conspicuity = self.get_intensity_conspicuity(image)
        color_conspicuity = self.get_color_conspicuity(image)
        orientation_conspicuity = self.get_orientation_conspicuity(image)

        # normalise and sum
        saliency = self.dog_normalization(intensity_conspicuity, 10) + \
                    self.dog_normalization(color_conspicuity, 10) + \
                    self.dog_normalization(orientation_conspicuity, 10)
        return saliency

    def small_to_large(self, small):
        # converts FOA location in the saliency image to FOA location in the original image
        converted = int(small[0] * 2 ** ((self.levels - 1) / 2)), int(small[1] * 2 ** ((self.levels - 1) / 2))
        return converted

    def draw_on_image(self, image, line=False):
        # line determines if line is drawn to connect to the last FOA location

        # convert FOA coordinates
        foa_converted = self.small_to_large(self.foa)
        last_foa_converted = self.small_to_large(self.last_foa)

        # draw a circle around FOA
        cv2.circle(image, foa_converted, self.radius, (255, 0, 0), 10)

        # join with lines
        if line:
            cv2.line(image, foa_converted, last_foa_converted, (0, 0, 255), 10)

    def update(self, image, inhibit=True):

        # increment the saliency map
        self.saliency_map += self.get_saliency(image)
        self.wta_map = np.copy(self.saliency_map)

        # see if threshold is reached
        foa_candidate = np.argwhere(self.wta_map >= self.wta_threshold)

        if foa_candidate.shape[0] > 0:
            # update FOA
            self.last_foa = self.foa
            indices = np.unravel_index(np.argmax(self.wta_map, axis=None), self.saliency_map.shape)
            self.foa = indices[1], indices[0]

            # reset all WTA-neurons
            self.wta_map.fill(0)

            # local inhibition of SM-neurons
            if inhibit:
                cv2.circle(self.saliency_map, self.foa, self.inhibition_radius, (0, 0, 0), -1)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # first get the shape of the frame
    ret, frame = cap.read()
    foa_finder = Saliency(frame.shape)

    plot = True
    if plot:
        # plot all the FOAs
        all_foa = np.zeros(frame.shape, dtype=np.uint8)

    # resize window
    ratio = frame.shape[0] / frame.shape[1]
    window_size = (int(1000 * ratio), 1000)
    window = cv2.namedWindow('Saliency 2000', cv2.WINDOW_NORMAL)
    if plot:
        window_size = window_size[0] * 2, window_size[0]
    cv2.resizeWindow('Saliency 2000', window_size)


    while True:
        # read image
        ret, frame = cap.read()

        # update the FOA location
        foa_finder.update(frame)

        # draw on the
        foa_finder.draw_on_image(frame, line=True)

        if plot:
            cv2.circle(all_foa, foa_finder.small_to_large(foa_finder.foa), 10, (0, 0, 255), -1)
            two_frames = np.hstack((frame, all_foa))
            cv2.resizeWindow('Saliency 2000', window_size)
            cv2.imshow('Saliency 2000', two_frames)

        else:
            cv2.imshow('Saliency 2000', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

