import sys
from cv2 import GaussianBlur
from skimage.io import imread, imsave

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    for i in range(200):
        image = imread('{}/{:03d}.png'.format(input_dir, i))
        imsave('{}/{:03d}.png'.format(output_dir, i), GaussianBlur(image, (5, 5), 0))
        print('it\'s processing the {}-th image'.format(i), end = '\n' if i == 199 else '\r')