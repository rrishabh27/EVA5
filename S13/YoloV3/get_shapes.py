# NEW FEATURES
# -------------------------------------------------------------------------------
# Name:        Outputting format for orders
# Purpose:     Outputting format for orders training yoloV3
# Author:      aka9
# Created:     28/07/2019

#
# -------------------------------------------------------------------------------

import glob
import os
import sys
from PIL import Image


def get_shapes():
    current_dir = 'D:\jupyter\YoloV3-master\data\customdata\images'
    # joinee = './data/customdata/'

    # Create and/or truncate train.txt and test.txt
    file_img_path_shapes = open('shapes.txt', 'w')

    # Populate train.txt and test.txt
    counter = 0

    print("Warning: we assume that all pictures to be processed are jpgs")
    for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
        #        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        im = Image.open(pathAndFilename) 
        sz = str(im.size)

        bad_chars = "(,)"
        for c in bad_chars:
            sz = sz.replace(c, "")

        file_img_path_shapes.write(sz + '\n')
        counter = counter + 1

    print(f'finished and number of images = {counter}')


def main():
    #array = sys.argv[1:]
    #if len(array) > 1: print("Not the right number of arguments, use -h for usage recommendations")
    # elif len(array) == 0: print("Not the right number of arguments, use -h for usage recommendations")
    # elif array[0] == '-h': print('simply write the path_to_folder as an argument')
    # else:
    #if not os.path.exists(array[0]): print('folder not found')
    # else: process(array[0])
    get_shapes()


if __name__ == '__main__':
    main()
