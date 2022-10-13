from __future__ import division
import argparse
import cv2
import math
import os
import shutil
import numpy as np
from collections import defaultdict


def declare_sorted_lines():
    global sorted_lines
    sorted_lines = {}


class DataAggregation(object):
    DATA = defaultdict(lambda: defaultdict(int))


class Environment(DataAggregation):
    def __init__(self):
        self.csv_meta_data = 'meta_data.csv'
        self.current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        self.source_folder = 'source_folder' + '/'
        self.destination_folder = 'cleaned_folder' + '/'
        self.extracted_folder = 'extracted_folder' + '/'

    @staticmethod
    def create_folder(folder):
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
            print ('Folder is created {0}'.format(folder))
            return folder
        except TypeError:
            exit('Directory cannot be created {0}. Execution is stopped!!!'.format(folder))

    @staticmethod
    def remove_file_from_folder(folder, file_name):
        try:
            os.remove(folder + file_name)
            print ("File {0} have been removed from folder {1}".format(file_name, folder))
        except OSError:
            print ("There is no file '{0}' in folder '{1}'".format(file_name, folder))

    def remove_failed_file_from_folder(self, failed_files):
        for key, file_name in enumerate(os.listdir(self.destination_folder)):
            if file_name.replace('_cleaned.png', '') in failed_files:
                self.remove_file_from_folder(folder=self.destination_folder, file_name=file_name)


class ImageProcessing(Environment):
    def __init__(self, erode_limit, harrison_corner):
        super(ImageProcessing, self).__init__()
        self.erode_limit = erode_limit
        self.harrison_corner = harrison_corner

    @staticmethod
    def read_source_file(source_file):
        _uploaded_source_file = cv2.imread(source_file)
        if _uploaded_source_file is None:
            exit("File '{0}' was not uploaded. Execution is stopped !!!".format(source_file))
        else:
            return _uploaded_source_file

    def get_text_harrison(self, original_file=None, output_extracted_file=None):

        # # outcome extracted file name
        # self.outcome_extracted_file = self.extracted_folder + output_extracted_file

        try:
            img_original_file = self.read_source_file(original_file)
            image = img_original_file.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)

            # Set Harris corners
            dst = cv2.cornerHarris(gray, self.harrison_corner, 31, 0.04)
            dst = cv2.dilate(dst, None)

            # Set inversion for threshold
            image[dst > 0.01 * dst.max()] = [0, 0, 0]
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

            # Get contours
            im2, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Extract contours area
            mask = np.zeros_like(img_original_file)
            for k, v in enumerate(cnts):
                cv2.drawContours(mask, cnts, k, (255, 255, 255), -1)
            out = np.zeros_like(img_original_file)
            out[:] = (255, 255, 255)
            out[mask == 255] = img_original_file[mask == 255]

            # save to file
            cv2.imwrite(output_extracted_file, out)

            print ('Image from: "{0}" is converted => "{1}"'.format(original_file, output_extracted_file))
        except cv2.error:
            exit('File {0} cannot be uploaded! Execution is  stopped !!!'.format(output_extracted_file))

    def get_text(self, original_file=None, output_extracted_file=None):
        # # outcome extracted file name
        # self.outcome_extracted_file = self.extracted_folder + output_extracted_file

        # convert file to gray scale
        try:
            img_original_file = self.read_source_file(original_file)
            gray = cv2.cvtColor(img_original_file, cv2.COLOR_BGR2GRAY)

            # compute the Scharr gradient magnitude representation of the images
            # in both the x and y direction
            grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

            # subtract the y-gradient from the x-gradient
            gradient = cv2.subtract(grad_x, grad_y)
            gradient = cv2.convertScaleAbs(gradient)

            # blur and threshold the image
            blurred = cv2.blur(gradient, (5, 5))
            (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

            # construct a closing kernel and apply it to the threshold image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (160, 37))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # perform a series of erosion and dilation's
            closed = cv2.erode(closed, None, iterations=self.erode_limit)
            closed = cv2.dilate(closed, None, iterations=36)

            # find the contours in the threshold image, then sort the contours
            # by their area, keeping only the largest one
            (_, contours, _) = cv2.findContours(closed.copy(), cv2.CHAIN_APPROX_SIMPLE,
                                                cv2.CHAIN_APPROX_SIMPLE)
            # create a mask with black background
            mask = np.zeros_like(img_original_file)
            for k, v in enumerate(contours):
                cv2.drawContours(mask, contours, k, (255, 255, 255), -1)

            out = np.zeros_like(img_original_file)
            out[:] = (255, 255, 255)

            out[mask == 255] = img_original_file[mask == 255]

            # save to file
            cv2.imwrite(output_extracted_file, out)

            print ('Image from: "{0}" is converted => "{1}"'.format(original_file, output_extracted_file))
        except cv2.error:
            exit('File {0} cannot be uploaded! Execution is  stopped !!!'.format(output_extracted_file))

    def clean_up(self, output_cleaned_file,
                 original_file,
                 adaptive_threshold=None,
                 binary_threshold=None,
                 small_contour_limit=None,
                 big_contour_limit=None,
                 vertical_limit=None,
                 horizontal_limit=None,
                 clean_image='clean'):

        # save original file
        img = self.read_source_file(original_file)

        # convert to gray and binaries image
        original_image = cv2.medianBlur(img.copy(), 5)
        original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # apply threshold to the image
        if not adaptive_threshold:
            try:
                _, image_after_threshold = cv2.threshold(original_image_gray, binary_threshold, 255, cv2.THRESH_BINARY)
            except cv2.error:
                image_after_threshold = None
                exit("Cannot clean image and im_bw variable is None")
        else:
            image_after_threshold = cv2.adaptiveThreshold(original_image_gray, 255,
                                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY,
                                                          blockSize=25,
                                                          C=3)
        # apply Gaussian blur to the image
        blur = cv2.GaussianBlur(image_after_threshold, (5, 5), 0)
        _, im_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im_bw = cv2.GaussianBlur(im_bw, (11, 11), 0)

        if clean_image == 'not-clean':
            # apply blur for cleaning image
            blur = cv2.GaussianBlur(image_after_threshold, (5, 5), 0)
            _, im_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blur = cv2.GaussianBlur(im_bw, (11, 11), 0)

            # save to file
            cv2.imwrite(output_cleaned_file, blur)
            print ('Image is not cleaned. Saved as -> "{0}"'.format(output_cleaned_file))
            return

        # run morphology filter to clean up image(only for adaptive threshold)
        if adaptive_threshold:
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(image_after_threshold, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            blur = cv2.GaussianBlur(closing, (5, 5), 0)
            erode = cv2.erode(blur, kernel, iterations=1)
            im_bw = cv2.adaptiveThreshold(erode, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # find contours
        (_, contours, _) = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)

        if small_contour_limit and big_contour_limit:
            # remove big and small contours
            cleaned_contours = [_ for _ in contours if small_contour_limit < cv2.contourArea(_) < big_contour_limit]

            # draw contours on the original threshold image
            cv2.drawContours(image_after_threshold, cleaned_contours, -1, (255, 255, 255), 3)

        if vertical_limit and horizontal_limit:
            # remove horizontal and vertical contours
            (_, contours, _) = cv2.findContours(image_after_threshold.copy(), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_NONE)
            cleaned_contours = [_ for _ in contours if
                                not self.contour_is_vertical(_, vertical_limit) and
                                not self.contour_is_horizontal(_, horizontal_limit)]

            # draw contours on the original threshold image
            cv2.drawContours(image_after_threshold, cleaned_contours, -1, (255, 255, 255), 3)

        # apply blur for cleaning image
        blur = cv2.GaussianBlur(image_after_threshold, (5, 5), 0)
        _, im_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(im_bw, (11, 11), 0)

        # save to file
        cv2.imwrite(output_cleaned_file, blur)
        print ('Cleaned image is saved as -> "{0}"'.format(output_cleaned_file))

    @staticmethod
    def contour_is_small(c=None, limit=50):
        if cv2.arcLength(c, closed=True) < limit:
            return True
        else:
            return False

    @staticmethod
    def resize_image_to_odd(source_image):
        height, width = source_image.shape
        if height % 2 == 1:
            height += 1
        if width % 2 == 1:
            width += 1
        return cv2.resize(source_image, (width, height))

    @staticmethod
    def contour_is_vertical(c=None, limit=4):
        bounding_box = cv2.boundingRect(c)
        contour_rank = bounding_box[3] / bounding_box[2]
        if contour_rank > limit and bounding_box[3] > 100:  # 100 is the size of line (everything above 100 is deleted)
            return True
        else:
            return False

    @staticmethod
    def contour_is_horizontal(c=None, limit=5):
        bounding_box = cv2.boundingRect(c)
        contour_rank = bounding_box[2] / bounding_box[3]
        return True if contour_rank < limit else False

    def remove_vertical_contour(self, cnt, limit=None):
        """
        Remove all vertical contours based on the high & width rank (recursively run till limit condition
        will \not be reached)
        :param cnt: list of contours
        :param limit: (high / width) rank
        :return: list of contours without contours which have h / w >= limit
        """
        contour_rank = {}

        # get data boxes for contours
        bounding_boxes = [cv2.boundingRect(c) for c in cnt]

        if bounding_boxes:
            # get high / width values
            for k, v in enumerate(bounding_boxes):
                contour_rank[k] = v[3] / v[2]

                max_index = max(contour_rank, key=contour_rank.get)

                sort_by_h_w_value = sorted(contour_rank, key=contour_rank.get)

                max_value = contour_rank[max_index]

                del cnt[sort_by_h_w_value[-1:][0]]

                if max_value >= limit and limit is not None:
                    return self.remove_vertical_contour(cnt, limit=limit)
                else:
                    return cnt
        else:
            print ("Seems image do not have vertical contours. Image can be corrupted !!!")
            return cnt

    @staticmethod
    def remove_small_contour(cnt, limit=None):
        """
        Remove all small contours based on the rectArea value and limit condition
        :param cnt: list of contours
        :param limit: contour size
        :return: list of contours without contours which have size =< limit
        """
        try:
            cnt_without_small_contours = [c for k, c in enumerate(cnt)
                                          if {k: cv2.arcLength(c, closed=True)}.values()[0] > limit]
            if not cnt_without_small_contours:
                print ("Seems image  do not have small contours. Image can be corrupted !!!")
            return cnt_without_small_contours if cnt_without_small_contours else cnt
        except TypeError or ValueError:
            exit("Image {0} is not ready for processing.")

    @staticmethod
    def get_x_y_for_moments(cnt):
        line_x = {}
        line_y = {}

        # all contour centers
        for k, v in enumerate(cnt):
            m = cv2.moments(v)
            centroid_x = int(m['m10'] / m['m00'])
            centroid_y = int(m['m01'] / m['m00'])
            line_y[k] = centroid_y
            line_x[k] = centroid_x
        return line_y, line_x

    @staticmethod
    def _merge_dict_with_identical_keys(x, y):
        return {key: (x[key], y[key]) for key in x}

    @staticmethod
    def _compare_tuples(tuple1, tuple2):
        return (np.array(tuple1) == tuple2).all()

    def _find_tuple_in_dict_of_tuples(self, tuple_to_find, list_of_tuples):
        # (1,2) in [(2,3), (1,2)] and return index of element
        result = [i[0] for i in list_of_tuples.items() if self._compare_tuples(i[1], tuple_to_find)]
        if result:
            return result
        else:
            return None

    def get_index_for_sorted_contours(self, moment_x, moment_y, moment_lines):
        ordered_contours = {}
        all_moments = self._merge_dict_with_identical_keys(moment_x, moment_y)

        for line_number, line in enumerate(moment_lines):
            ordered_contours[line_number] = []
            current_line = self._merge_dict_with_identical_keys(line[1], line[0])
            for point in current_line.values():
                ordered_contours[line_number].append(self._find_tuple_in_dict_of_tuples(point, all_moments)[0])

    def generate_moment_lines(self, delta=None, moment_x=None, moment_y=None, min_y=None, max_y=None, line_counter=0):
        if min_y is None and max_y is None:
            global_min_y = moment_y[min(moment_y, key=moment_y.get)]
            global_max_y = moment_y[max(moment_y, key=moment_y.get)]
            delta_range = range(int(math.fabs(global_min_y - delta)), global_min_y + delta)
        else:
            global_min_y = min_y
            global_max_y = max_y
            delta_range = range(global_min_y, global_min_y + delta)

        line_y_values = list(set(delta_range) & set(moment_y.values()))
        line_indexs = [moment_y.keys()[moment_y.values().index(v)] for v in line_y_values]
        line_x_values = {k: moment_x[k] for k in line_indexs}
        line_sorted_by_x = sorted(line_x_values, key=line_x_values.get)
        min_y = max(delta_range)
        max_y = global_max_y

        sorted_lines[line_counter] = line_sorted_by_x

        line_counter += 1

        if min_y <= max_y:
            self.generate_moment_lines(delta=60, moment_x=moment_x, moment_y=moment_y, min_y=min_y, max_y=max_y,
                                       line_counter=line_counter)

    def split_into_contours_(self, source_file, image_pattern_for_file='roi', folder=None):
        # income image
        img = cv2.imread(source_file)

        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('float32')
        gray /= 255

        # check for odd H and W of gray image (for performing dct() function )
        gray = self.resize_image_to_odd(source_image=gray)

        dct = cv2.dct(gray)
        vr = 1.  # vertical ratio
        hr = .95  # horizontal
        dct[0:vr * dct.shape[0], 0:hr * dct.shape[1]] = 0
        gray = cv2.idct(dct)
        gray = cv2.normalize(gray, -1, 0, 1, cv2.NORM_MINMAX)
        gray *= 255
        gray = gray.astype('uint8')

        # morphology function (defining size of contours)
        gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                                iterations=1)

        gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                                iterations=1)

        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reversed_arr = contours[::-1]
        sorted(reversed_arr, key=cv2.boundingRect, reverse=True)
        contours = reversed_arr

        contours = self.remove_vertical_contour(contours, limit=4)
        contours = self.remove_small_contour(contours, limit=50)

        moment_y, moment_x = self.get_x_y_for_moments(contours)

        self.generate_moment_lines(moment_x=moment_x, moment_y=moment_y, delta=60)

        line = 0
        for key, indices in sorted_lines.items():

            if indices:
                image_in_line = 0
                for indice in indices:
                    con = contours[indice]
                    cv2.drawContours(gray, con, -1, (255, 255, 255), 3)

                    boxmask = np.zeros(gray.shape, gray.dtype)

                    x, y, w, h = cv2.boundingRect(con)
                    cv2.rectangle(boxmask, (x, y), (x + w, y + h), color=255, thickness=-1)

                    roi = img[y:y + h, x:x + w]

                    if folder is None:
                        cv2.imwrite(image_pattern_for_file + str(line) + '_' + str(image_in_line) + ".png", roi)
                        image_in_line += 1
                    else:
                        cv2.imwrite(folder + "/" + "image_pattern_for_file" + str(line) + '__' + str(
                            image_in_line) + ".png", roi)
                        image_in_line += 1

                    cv2.imwrite('contours.jpg', img & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))

                print ('Line ' + str(line) + ' have ' + str(image_in_line) + 'contours')

            line += 1
        return line

    def extract_file_to_folder(self, image, folder_name):

        # create folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # copy image
        shutil.copy2(image, folder_name)

        # split image
        self.split_into_contours_(source_file=image, image_pattern_for_file=folder_name + '_', folder=folder_name)


class PostImageProcessing(Environment):
    def __init__(self):
        Environment.__init__(self)
        self.hist_comparison_file = self.current_dir + '/' + 'HIST_CLEANED.png'

    def get_file_from_dir(self):
        return [self.destination_folder + f for f in os.listdir(self.destination_folder)]

    @staticmethod
    def image_histograme_intersection(first_image, second_image):
        """
        Compare Histogram correlation of two images.
        :param first_image: properly cleaned image
        :type second_image: object
        """
        min_correlation_value = 0.99969000000

        open_cv_methods = {"Correlation": cv2.HISTCMP_CORREL,
                           "Chi-Squared": cv2.HISTCMP_CHISQR,
                           "Intersection": cv2.HISTCMP_INTERSECT,
                           "Hellinger": cv2.HISTCMP_BHATTACHARYYA}

        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
        first_hist = cv2.calcHist([first_image], [0, 1, 2], None, [8, 8, 8],
                                  [0, 256, 0, 256, 0, 256])
        cv2.normalize(first_hist, dst=first_hist).flatten()

        second_hist = cv2.calcHist([second_image], [0, 1, 2], None, [8, 8, 8],
                                   [0, 256, 0, 256, 0, 256])
        cv2.normalize(second_hist, dst=second_hist).flatten()

        result_hist_comparison = cv2.compareHist(first_hist, second_hist, open_cv_methods["Correlation"])

        if result_hist_comparison <= min_correlation_value:
            return False
        else:
            return True

    @staticmethod
    def image_color_intensity(image):
        """
        Check image intensity
        :param image:
        :return:
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        no_rows = gray_image.shape[0]
        no_cols = gray_image.shape[1]
        sumall = 0.0
        # Iterate through all pixels and compute the sum of their intensities
        for row in range(no_rows):
            for col in range(no_cols):
                sumall += float(gray_image[row, col])
        average = sumall / float(no_rows * no_cols)

        # Scale result to the interval [0,1] and print it
        return average / 255.0

    def define_quality_of_image(self):
        failed_f = []
        success_f = []
        for k in self.DATA.keys():
            if self.DATA[k]['contours_number'] == False or self.DATA[k]['intensity_value'] < 0.6:
                failed_f.append(self.get_file_name_from_path(k))
            else:
                success_f.append(self.get_file_name_from_path(k))
        return failed_f, success_f

    def set_img_parameters(self, cleaned_image_hist):
        print ('=' * 100)
        image_cleaned = cv2.imread(cleaned_image_hist, 1)
        for f in self.get_file_from_dir():
            image = cv2.imread(f, 1)
            hist = self.image_histograme_intersection(image_cleaned, image)
            intensity_value = self.image_color_intensity(image)

            if 'contours_number' not in self.DATA[self.current_dir + f].keys():
                self.DATA[self.current_dir + f]['contours_number'] = True
            self.DATA[self.current_dir + f]['histogram'] = hist
            self.DATA[self.current_dir + f]['intensity_value'] = intensity_value
        print ('Result have been generated for {0} items.'.format(len(self.DATA)))

    @staticmethod
    def get_file_name_from_path(file_path):
        return file_path.split('/')[-1:][0].replace('_cleaned.png', '')


def create_parser():
    parser = argparse.ArgumentParser(description='Setting parameters for cleaning image')

    parser.add_argument('-bt', '--binary_threshold', type=int,
                        help='set binary threshold value for image cleaning [50-150]', default=110)

    parser.add_argument('-sc', '--small_contour_limit', type=int,
                        help='Size of contour which will be removed from image [200-400]', default=250)
    parser.add_argument('-bc', '--big_contour_limit', type=int,
                        help='Size of contour which will be removed from image [10000-25000]', default=10000)

    parser.add_argument('-vc', '--vertical_contour_limit', type=int,
                        help='correlation between hight and wight of vertical contour [3-6]', default=4)
    parser.add_argument('-hc', '--horizontal_contour_limit', type=int,
                        help='correlation between height and wight of horizontal contour [3-6]', default=4)

    parser.add_argument('-el', '--erode_limit', type=int, help='Limit for extracting text from the image [12-24-48]',
                        default=14)
    parser.add_argument('-hl', '--harrison_limit', type=int, help='Harrison limitation [12-24-48]',
                        default=270)

    parser.add_argument('--adaptive_threshold', dest='adaptive_threshold', action='store_true',
                        help='Use adaptive or binary threshold [True / False]')
    parser.add_argument('--no-adaptive_threshold', dest='adaptive_threshold', action='store_false',
                        help='Use binary threshold')

    parser.add_argument('-tt', '--text_type', type=str, help='Get text type [edge-harrison]', default='harrison')
    parser.add_argument('-src', '--source_file', type=str, help='Path to the source file')

    parser.add_argument('-ci', '--clean_image', type=str, help='clean or not-clean', default='clean')

    cur_args = parser.parse_args()
    return cur_args


if __name__ == "__main__":
    args = create_parser()
    CLEAN = args.clean_image
    EDGE_LIMIT = args.erode_limit  # 14
    HARRISON_CORNER = args.harrison_limit  # 200 - 400
    GET_TEXT_TYPE = args.text_type  # 'edge' or 'harrison'
    ADAPTIVE_THRESHOLD = args.adaptive_threshold  # False or True
    BINARY_THRESHOLD = args.binary_threshold  # 90 - 140
    SMALL_CONTOUR_LIMIT = args.small_contour_limit  # 250-400
    BIG_CONTOUR_LIMIT = args.big_contour_limit  # 10000
    VERTICAL_LIMIT = args.vertical_contour_limit  # 3-5
    HORIZONTAL_LIMIT = args.horizontal_contour_limit  # 4-5
    ORIGINAL_FILE = args.source_file

    ip = ImageProcessing(erode_limit=EDGE_LIMIT, harrison_corner=HARRISON_CORNER)
    ip.harrison_corner = HARRISON_CORNER

    if GET_TEXT_TYPE == 'edge':
        print('EDGE get text...')
        ip.get_text(original_file=ORIGINAL_FILE,
                    output_extracted_file=ORIGINAL_FILE + '_extracted.png')
    elif GET_TEXT_TYPE == 'harrison':
        print('HARRISON get text...')
        ip.get_text_harrison(
            original_file=ORIGINAL_FILE,
            output_extracted_file=ORIGINAL_FILE + '_extracted.png')
    else:
        exit("'{0}' is not present".format(GET_TEXT_TYPE))

    ip.clean_up(original_file=ORIGINAL_FILE + '_extracted.png',
                output_cleaned_file=ORIGINAL_FILE + '_cleaned.png',
                adaptive_threshold=ADAPTIVE_THRESHOLD,
                binary_threshold=BINARY_THRESHOLD,
                small_contour_limit=SMALL_CONTOUR_LIMIT,
                big_contour_limit=BIG_CONTOUR_LIMIT,
                vertical_limit=VERTICAL_LIMIT,
                horizontal_limit=HORIZONTAL_LIMIT,
                clean_image=CLEAN,
                )
