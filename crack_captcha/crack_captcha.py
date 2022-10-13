# !/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import cv2

__author__ = 'Roman Byelyy'


class CrackCaptcha(object):
    def __init__(self, input_image, output_image):
        self.logger = logging.getLogger('captcha')
        self.original_image = input_image
        self.output_file = output_image

    # def _upload_image(self):
    #     self.logger.info('File {0} have been uploaded for analysis'.format(self.original_image))
    #     image = cv2.imread(self.original_image)
    #     image = CrackCaptcha.crop_captcha(image)
    #     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _upload_image(self):
        self.logger.info('File {0} have been uploaded for analysis'.format(self.original_image))
        try:
            img = cv2.imread(self.original_image)
        except OSError as e:
            self.logger.error(e)
            exit()
        else:
            return img

    def highlight_text(self):
        img = self._upload_image()
        img_ocean_color_map = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
        img_bone_color_map = cv2.applyColorMap(img_ocean_color_map, cv2.COLORMAP_BONE)
        _, img_th_15 = cv2.threshold(img_bone_color_map, 15, 255, cv2.THRESH_BINARY)
        _, img_th_110 = cv2.threshold(img_th_15, 110, 255, cv2.THRESH_BINARY)
        return img_th_110

    def save_cleaned_image(self):
        cleaned_image = self.highlight_text()
        cv2.imwrite('cleaned_captcha.jpg', cleaned_image)

    @staticmethod
    def crop_captcha(img):
        return img[330:365, 470:570]

    def execute_tesseract(self):
        os.system('tesseract cleaned_captcha.jpg outputbase_' + self.output_file + ' -l eng -psm 6')
        with open('outputbase_' + self.output_file + '.txt', 'r') as result_file:
            data = result_file.read().replace('\n', '')
        self.logger.info('Tesseract have been applied to file. CAPTCHA = "{0}"'.format(data))
        return data

    def __del__(self):
        try:
            os.remove('cleaned_captcha.jpg')
            os.remove('outputbase_' + self.output_file + '.txt')
        except OSError as e:
            print (e)

if __name__ == "__main__":
    crack_captcha = CrackCaptcha(
        input_image='../data/captcha000.jpg',
        output_image='output.jpg')

    crack_captcha.highlight_text()
    crack_captcha.save_cleaned_image()
    print (crack_captcha.execute_tesseract())

