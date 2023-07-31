# source: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

# to install opencv2:
# $ pip install opencv-python
import cv2
import argparse
import datetime
from typing import Optional, Tuple
import filecmp
from skimage.metrics import structural_similarity as ssim
import imutils
import os
import numpy as np
from pathlib import Path
from typing import Union

__version__ = '0.0.1'

from scrnshtr.util import trim_image

MIN_FLOAT_VAL = 0
MAX_FLOAT_VAL = 1


def main(dirs: Tuple[Path, Path], out: Optional[Path] = None, ia: bool = True, ib: bool = True,
         ma: bool = True, mb: bool = True, ot: bool = False, od: bool = False, maxssim: float = 0.99999,
         trim_images: bool = False, dir_suffix: str = '', trim_inputs: Optional[Tuple[int, int, int, int]] = None):
    dir1, dir2 = [Path(d) for d in dirs]
    if out is None:
        out = dir1.parent
    out = Path(out, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_visreg' + (
        '_' + dir_suffix if dir_suffix != '' else ''))

    if not out.exists():
        out.mkdir(exist_ok=True, parents=True)

    list_of_allowed_extensions = ['.png']
    list_of_filenames = []
    list_of_files_1 = [entry for entry in os.listdir(dir1) if os.path.splitext(entry)[1] in list_of_allowed_extensions]
    list_of_files_2 = [entry for entry in os.listdir(dir2) if os.path.splitext(entry)[1] in list_of_allowed_extensions]
    files_intersec = sorted(list(set(list_of_files_1).intersection(list_of_files_2)))

    print(f'not in final set (difference): '
          f'{set(list_of_files_1 + list_of_files_2).symmetric_difference(files_intersec)}')
    trim_inputs = (0, 30, 0, 0)
    for file in sorted(list(files_intersec)):
        filename1 = os.path.join(dir1, file)
        filename2 = os.path.join(dir2, file)
        if not filecmp.cmp(f1=filename1, f2=filename2, shallow=False):
            list_of_filenames.append(file)
            print(filename1, filename2)
            try:
                OpenCVDiff(file1=str(filename1), file2=str(filename2), output_dir=out,
                           output_diff=od, output_threshold=ot,
                           output_image_a=ma, output_input_image_a=ia,
                           output_input_image_b=ib, output_image_b=mb,
                           max_ssim=maxssim, trim_images=trim_images, trim_inputs=trim_inputs)
            except ValueError as err:
                print(err)
                print(file)
        else:
            print(f'files identical: "{os.path.split(filename1)[1]}"')


def add_suffix(filename: Union[Path, str], suffix: str) -> str:
    return os.path.join(os.path.splitext(filename)[0] + suffix + os.path.splitext(filename)[1])


def resize_and_pad(img: np.ndarray, size: tuple, pad_color=0) -> np.ndarray:
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    ratiow = sw / w
    ratioh = sh / h

    if ratioh > ratiow:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif ratioh < ratiow:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(pad_color,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=pad_color)
    return scaled_img


class OpenCVDiff:
    # ToDo: implement horizontal concat of before & after images via Pillow
    # ToDo: threshold as parameter
    def __init__(self, file1: Union[Path, str],
                 file2: Union[Path, str],
                 output_dir: Union[Path, str],
                 max_ssim: float,
                 output_diff: bool = True,
                 output_threshold: bool = True,
                 output_input_image_a: bool = True,
                 output_input_image_b: bool = True,
                 output_image_a: bool = True,
                 output_image_b: bool = True,
                 trim_images: bool = False,
                 trim_inputs: Optional[Tuple[int, int, int, int]] = None
                 ):
        """

        :type trim_images: trim borders tuple: (t, b, l, r)
        """
        # load the two input images
        self.imageA = cv2.imread(file1)
        self.imageB = cv2.imread(file2)

        if trim_inputs is not None:
            self.imageA = self.imageA[trim_inputs[0]:self.imageA.shape[0]-trim_inputs[1], trim_inputs[2]:self.imageA.shape[1]-trim_inputs[3]].copy()
            self.imageB = self.imageB[trim_inputs[0]:self.imageB.shape[0]-trim_inputs[1], trim_inputs[2]:self.imageB.shape[1]-trim_inputs[3]].copy()

        self.imageA_input = self.imageA.copy()
        self.imageB_input = self.imageB.copy()

        h_a, w_a = self.imageA.shape[:2]
        h_b, w_b = self.imageB.shape[:2]

        max_h = max(h_a, h_b)
        max_w = max(w_a, w_b)
        if h_a != max_h or w_a != max_w:
            self.imageA = resize_and_pad(self.imageA, (max_h, max_w)).copy()
        if h_b != max_h or w_b != max_w:
            self.imageA = resize_and_pad(self.imageB, (max_h, max_w)).copy()

        # convert the images to grayscale
        self.grayA = cv2.cvtColor(self.imageA, cv2.COLOR_BGR2GRAY)
        self.grayB = cv2.cvtColor(self.imageB, cv2.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = ssim(self.grayA, self.grayB, full=True)
        diff = (diff * 255).astype("uint8")

        if score > max_ssim:
            print(f'file "{os.path.split(file1)[1]}": SSIM {score} > {max_ssim}: too similar, will NOT be processed')
            return
        else:
            print(f'file "{os.path.split(file1)[1]}": SSIM {score} <= {max_ssim}: PROCESSING')

            # do absdiff and threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            absdiff = cv2.absdiff(self.grayA, self.grayB)
            thresh = cv2.multiply(absdiff, 5)
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            # loop over the contours
            for c in contours:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(self.imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(self.imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # show the output images

            output_paths = []
            output_base_path = Path(output_dir, os.path.split(file1)[1])

            if output_input_image_a:
                output_path = add_suffix(output_base_path, '_A_in')
                cv2.imwrite(output_path, self.imageA_input)
                output_paths.append(output_path)
            if output_input_image_b:
                output_path = add_suffix(output_base_path, '_B_in')
                cv2.imwrite(output_path, self.imageB_input)
                output_paths.append(output_path)
            if output_image_a:
                output_path = add_suffix(output_base_path, '_A_mod')
                cv2.imwrite(output_path, self.imageA)
                output_paths.append(output_path)
            if output_image_b:
                output_path = add_suffix(output_base_path, '_B_mod')
                cv2.imwrite(output_path, self.imageB)
                output_paths.append(output_path)

            if output_diff:
                output_path = add_suffix(output_base_path, '_diff')
                cv2.imwrite(output_path, diff)
                output_paths.append(output_path)
            if output_threshold:
                output_path = add_suffix(output_base_path, '_thresh')
                cv2.imwrite(output_path, thresh)
                output_paths.append(output_path)

            if trim_images:
                for f in output_paths:
                    trim_image(Path(f))
                # [trim_image(Path(f)) for f in output_paths]


def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN_FLOAT_VAL or f > MAX_FLOAT_VAL:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX_FLOAT_VAL) + "and > " + str(MIN_FLOAT_VAL))
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool for image comparison")
    parser.add_argument('dirs', nargs=2, help='input dirs')
    parser.add_argument('-o', '--out', help='output path')
    parser.add_argument('-ia', help='output: input image a', action='store_true')
    parser.add_argument('-ib', help='output: input image b', action='store_true')
    parser.add_argument('-ma', help='output: modified image a', action='store_true')
    parser.add_argument('-mb', help='output: modified image b', action='store_true')
    parser.add_argument('-od', help='output: diff', action='store_true')
    parser.add_argument('-ot', help='output: threshold', action='store_true')
    parser.add_argument('-maxssim', help='set max ssim (similarity) value to process', type=range_limited_float_type,
                        default=0.99999)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args()
    args.dirs = tuple(args.dirs)

    main(**args.__dict__)
