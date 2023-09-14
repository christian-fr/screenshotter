import datetime
from pathlib import Path
import os
from typing import List, Tuple
from PIL import Image, ImageChops, ImageDraw, ImageOps
import unicodedata
import re

from PIL import ImageFont


def flatten(ll: list) -> list:
    return [a for e in ll for a in e]


def trim(im: Image) -> Image:
    bg = Image.new(mode=im.mode, size=im.size, color=im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        # Failed to find the borders, convert to "RGB"
        return trim(im.convert('RGB'))


def trim_image(image_file: Path, out_path: Path = None) -> None:
    if out_path is None:
        out_path = image_file
    trim(Image.open(image_file)).save(out_path)


def slugify(value, allow_unicode=False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def check_for_files(output_path: Path) -> None:
    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        print(f'output dir ({output_path=}) exists.\n')

        list_of_files = [Path(output_path, str(entry)) for entry in os.listdir(output_path) if
                         Path(output_path, str(entry)).is_file()]
        if list_of_files:
            print(f'output dir ({output_path=}) is not empty:\n')

            print('\n'.join([str(entry) for entry in list_of_files]))
            choice = input('delete those files? ([y/n]')
            if choice.strip() in ['y', 'Y']:
                [os.remove(entry) for entry in list_of_files]


def add_str_to_image(file: Path, text: str, color: str, location: str, size: int = 18, rotate: int = 0,
                     margin: int = 5):
    assert rotate in [0, 90, 180, 270]
    assert location in ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw']
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 16)

    image = Image.open(file)
    draw = ImageDraw.Draw(image)

    txt = Image.new('L', font.getsize(text))
    d = ImageDraw.Draw(txt)
    d.text((0, 0), text, font=font, fill=255)
    w = txt.rotate(rotate, expand=True)
    text_width, text_height = w.size

    im_width, im_height = image.size
    if 'w' in location:
        x = margin
    elif 'e' in location:
        x = im_width - text_width - margin
    else:
        x = int(round(im_width / 2)) - int(round(text_width / 2))

    if 'n' in location:
        y = margin
    elif 's' in location:
        y = im_height - text_height - margin
    else:
        y = int(round(im_height / 2)) - int(round(text_height / 2))

    image.paste(ImageOps.colorize(w, "black", color), (x, y), w)
    image.save(file)


def prepare_images(zipped_files_list: List[Tuple[Path, Path]], output_path: Path, trim_result: bool = True) -> List[
    Path]:
    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)
    list_of_output_files = []
    for index, files_tuple in enumerate(zipped_files_list):
        print(index)
        filename_dir1 = files_tuple[0].name
        filename_dir2 = files_tuple[1].name

        page_name_dir1 = filename_dir1[:filename_dir1.find('##')]
        page_name_dir2 = filename_dir1[:filename_dir2.find('##')]
        assert page_name_dir1 == page_name_dir2

        output_filename = Path(output_path,
                               f'{str(index).zfill(len(str(len(zipped_files_list))))}__{page_name_dir1}.png')
        list_of_output_files.append(output_filename)
        get_concat_h(Image.open(files_tuple[0]), Image.open(files_tuple[1])).save(str(output_filename))
        print(f'written file: {output_filename=}')
        if trim_result:
            trim_image(output_filename, output_filename)
    return list_of_output_files


def remove_files(list_of_files: List[Path]) -> None:
    choice = input('remove files? [y/n]')
    if choice.strip() in ['y', 'Y']:
        [os.remove(file) for file in list_of_files]


def get_concat_v(im1, im2) -> Image:
    dst = Image.new('RGB', (im1.width, im1.scroll_height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def get_concat_h(im1, im2) -> Image:
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def list_collector(output_file: Path, token: str, page: str, input_list: List[str], lang: str) -> None:
    if not output_file.exists():
        out_str = ''
    else:
        out_str = output_file.read_text(encoding='utf-8')
        out_str += f'\n\n'
    out_str += '# ' + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f') + '\n'
    input_list = [f'{token}\t{page}\t{lang}\t{s}' for s in input_list]
    out_str += '\n'.join(input_list)
    output_file.write_text(out_str, encoding='utf-8')
