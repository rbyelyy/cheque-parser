import argparse
import cv2
import numpy as np
import math
import os


def declare_sorted_lines():
    global sorted_lines
    sorted_lines = {}


def remove_small_contour(cnt, limit=None):
    """
    Remove all small contours based on the rectArea value and limit condition
    :param cnt: list of contours
    :param limit: contour size
    :return: list of contours without contours which have size =< limit
    """
    contour_rank = {}

    # get data boxes for contours
    length = [cv2.arcLength(c, closed=True) for k, c in enumerate(cnt)]
    for k, v in enumerate(length):
        contour_rank[k] = v

    min_index = min(contour_rank, key=contour_rank.get)
    min_value = contour_rank[min_index]

    sort_by_size = sorted(contour_rank, key=contour_rank.get)

    del cnt[sort_by_size[:1][0]]

    if min_value <= limit and limit is not None:
        return remove_small_contour(cnt, limit=limit)
    else:
        return cnt


def get_x_y_for_moments(cnt):
    line_x = {}
    line_y = {}

    # all contour centers
    for k, v in enumerate(cnt):
        m = cv2.moments(v)
        if m['m00'] == 0:
            continue
        centroid_x = int(m['m10'] / m['m00'])
        centroid_y = int(m['m01'] / m['m00'])
        line_y[k] = centroid_y
        line_x[k] = centroid_x
    return line_y, line_x


def generate_moment_lines(delta=None, moment_x=None, moment_y=None, min_y=None, max_y=None, line_counter=0):
    if min_y is None and max_y is None:  # first time run
        global_min_y = moment_y[min(moment_y, key=moment_y.get)]
        global_max_y = moment_y[max(moment_y, key=moment_y.get)]
        delta_range = range(int(math.fabs(global_min_y - delta)), global_min_y + delta)
    else:  # second time run
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
        generate_moment_lines(delta=60, moment_x=moment_x, moment_y=moment_y, min_y=min_y, max_y=max_y,
                              line_counter=line_counter)


def detect_text_area(img, cur_granularity=13):
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(img, kernel, iterations=cur_granularity)
    #erode = cv2.cvtColor(erode, cv2.CV_32SC1)
    return erode


def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnt, hnt = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # get contours
    hnt = hnt[0]
    return cnt, hnt


def remove_nested_contours(cnt):
    contours_to_remove = []
    for index, value in enumerate(hierarchy):
        if any(value[2]) != -1:
            contours_to_remove.append(index)

    for index, value in enumerate(cnt):
        if index in contours_to_remove:
            del cnt[index]
    return cnt


def draw_contours(img, cnt, limit_for_small=70):
    # for each contour found, draw a rectangle around it on original image
    for index, contour in enumerate(cnt):
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        if h < limit_for_small or w < limit_for_small:
            del cnt[index]

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # write original image with added contours to disk
        cv2.imwrite("contoured.jpg", img)


def is_image_white(img, pix_limit):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    black_scene = sum([v for k, v in enumerate(hist) if k < 50])
    white_scene = sum([v for k, v in enumerate(hist) if k > 200])

    white_pxs = white_scene[0]
    black_pxs = black_scene[0]

    return True if (black_pxs < 200) else False


def is_contour_horizontal(cnt, limit=None):
    bounding_box = cv2.boundingRect(cnt)
    contour_rank = bounding_box[2] / bounding_box[3]
    if contour_rank > limit:
        return True
    else:
        return False


def sort_by_moment(cnt):
    # lets do some sorting here
    moment_y, moment_x = get_x_y_for_moments(cnt)
    generate_moment_lines(moment_x=moment_x, moment_y=moment_y, delta=60)


def split_and_save(folder=None, file_pattern='line_'):
    # create a folder if not exists
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # number of lines in bill
    all_lines = len(sorted_lines)
    s = sorted_lines

    for line_number in range(all_lines):
        contours_to_display = []
        [contours_to_display.append(contours[contour_index]) for contour_index in sorted_lines[line_number] if
         sorted_lines[line_number]]
        if contours_to_display:
            for serial_number, c in enumerate(contours_to_display):
                if c is not None:
                    x, y, w, h = cv2.boundingRect(c)
                    roi = image[y:y + h, x:x + w]

                    if not is_image_white(img=roi, pix_limit=3):
                        cv2.imwrite(folder + '/' + file_pattern + '{0}_part_{1}.png'.format(line_number, serial_number),
                                    roi)


def create_parser():
    parser = argparse.ArgumentParser(description='Limitation for small images (default 70) and granularity of contours')

    parser.add_argument('-g', '--granularity', type=str,
                        help='specify granularity of contours for splitting [low-medium-high]',
                        required=True)
    parser.add_argument('-d', '--directory', type=str, help='folder for storing splitted images')
    parser.add_argument('-f', '--file', type=str, help='file name for spliting')
    cur_args = parser.parse_args()
    if cur_args.granularity == 'high':
        current_granularity = 1100
    elif cur_args.granularity == 'medium':
        current_granularity = 700
    elif cur_args.granularity == 'low':
        current_granularity = 400
    else:
        current_granularity = None
        exit('Please set granularity as "high/medium/low"')

    return cur_args, current_granularity


if __name__ == "__main__":
    granularity_status = {'low': 15, 'medium': 5, 'high': 1}
    args, granularity = create_parser()
    cur_gran = args.granularity
    declare_sorted_lines()

    image = cv2.imread(args.file, 0)

    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    if cur_gran != 'high':
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=granularity_status[cur_gran])

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print 'Initial number of contours: ' + str(len(contours))
    contours = remove_nested_contours(cnt=contours)
    print 'After removing nested contours: ' + str(len(contours))
    try:
        contours = remove_small_contour(cnt=contours, limit=100)
        print 'After removing small contours: ' + str(len(contours))
    except RuntimeError:
        print 'Its too many trash on the image so it is hard to remove all contours. '

    sort_by_moment(contours)
    split_and_save(folder=args.directory)

















