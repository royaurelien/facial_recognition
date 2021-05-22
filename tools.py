import csv
import os
import uuid
from face_recognition.api import face_locations
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import logging
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import hashlib
import dlib

import face_recognition as face

_logger = logging.getLogger(__name__)

def hash_array(numpy_array):
    return int(hashlib.md5(bytes(numpy_array)).hexdigest(), base=16)

def scan_folder(path, extension=[]):
    files = []
    for root, subdirs, current_files in os.walk(path):
        files += [os.path.join(root,name) for name in current_files if name.endswith(tuple(extension))]

    return files

def resize(picture, max_width=1600):
    #Scale down image if large, performance suggestion snippet from face_recognition
    if picture.shape[1] > max_width:
        scale_factor = float(max_width) / picture.shape[1]
        im = Image.fromarray(picture)
        size = tuple((np.array(im.size) * scale_factor).astype(int))
        picture = np.array(im.resize(size))

        _logger.debug("\tResizing to {size[0]}x{size[1]} pixels with {o:.2f}x".format(size=size, o=scale_factor))

    return picture

def new_person(path, prefix='person'):
    if not os.path.isdir(path):
        os.mkdir(path)

    dirs = os.listdir(path)
    if dirs:
        index = sorted([int(elem.split('_')[1]) for elem in dirs])[-1]
        index += 1
    else:
        index = 1

    name = "_".join([prefix,str(index)])
    os.mkdir(os.path.join(path,name))

    return name

def generate_id(**kwargs):
    return  uuid.uuid4()

def save_array(path, filename, array):
    if not os.path.isdir(path):
        os.mkdir(path)

    filepath = os.path.join(path, "{}.npy".format(filename))
    encoded_file = np.save(filepath, array)

def mask_circle_transparent(pil_img, blur_radius, offset=0):
    offset = blur_radius * 2 + offset
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    result = pil_img.copy()
    result.putalpha(mask)
    result = result.resize((64,64), Image.LANCZOS)

    return result

def save_thumbnail(path, filename, image, pos=(), extension='jpg', padding=0, crop='square'):
    if not os.path.isdir(path):
        os.mkdir(path)

    filepath = os.path.join(path, "{}.{}".format(filename, extension))

    if padding:
#         pos = (left, top, right, bottom)
        pos = list(pos)
        pos[0] -= padding
        pos[1] -= padding
        pos[2] += padding
        pos[3] += padding
        pos = tuple(pos)

    new_image = image.crop(pos)

    if crop == 'circle':
        new_image = mask_circle_transparent(new_image, 0)

    new_image.save(filepath, quality=90)

def compress_and_save(path, list_of_key, list_of_array, delete=False):
    data = {key:array for key, array in zip(list_of_key, list_of_array)}
    np.savez_compressed(path, **data)

def load_from_npz(path):
    if not os.path.isfile(path):
        _logger.warning('File not found: {}'.format(path))
        return [], [], []

    tmp = np.load(path)
    # hash_list = [hash_array(array) for array in tmp.values()]
    # return list(tmp.keys()), hash_list, list(tmp.values())

    return [], [int(k) for k in tmp.keys()], list(tmp.values())

def load_encodings(path, npz_file=None):
    if npz_file is None:
        # try to load npi files from path
        raise NotImplementedError
    else:
        filepath = os.path.join(path, npz_file)
        names, hash_list, encodings = load_from_npz(filepath)

    return names, hash_list, encodings

def linear_equation(series):
    x = [serie[0] for serie in series]
    y = [serie[1] for serie in series]
    min_value, max_value = min(x),max(x)

    return Polynomial.fit(x, y, 1, window=[min_value,max_value])

#     x = [0.1, 0.55]
#     y = [90, 50]

def polynomial_equation(series):
    x = [serie[0] for serie in series]
    y = [serie[1] for serie in series]
    min_value, max_value = min(x),max(x)

    return Polynomial.fit(x, y, 1, window=[min_value,max_value])

def read_csv(path, filter_key=None):
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        if filter_key:
            lines = [row for row in reader if row[filter_key]]
        else:
            lines = [row for row in reader]

    for line in lines:
        if 'save' in line:
            line['save'] = eval(line['save'])

    return lines

def save_csv(path, fieldnames, lines, append=True):
    create_file = True if not os.path.isfile(path) else False
    mode = 'a' if append else 'w'

    with open(path, mode) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if create_file:
            writer.writeheader()

        for line in lines:
            writer.writerow(line)

def process_file(path, detector, **options):

    upsample = options.get('upsample', 1)

    logging.info(">> {}".format(path))

    # picture = dlib.load_rgb_image(path)

    picture = face.load_image_file(path)
    picture = resize(picture)

    # search for face locations
    face_locations = face.face_locations(picture,number_of_times_to_upsample=upsample)
    # face_locations = detector(picture, upsample)
    pil_image = Image.fromarray(picture)
    md5 = int(hashlib.md5(pil_image.tobytes()).hexdigest(), base=16)
    # face_locations = []
    del pil_image

    logging.debug("\tCalculating md5 hash: {}".format(md5))

    return picture, md5, face_locations


def multi_process_file(files, detector, options):
    images, list_of_hash, list_of_locations = [], [], []

    for filename in files:
        image_array, image_hash, face_locations = process_file(filename, detector, **options)

        images.append(image_array)
        list_of_hash.append(image_hash)
        list_of_locations.append(face_locations)
        del image_array
        del image_hash
        del face_locations

    return images, list_of_hash, list_of_locations