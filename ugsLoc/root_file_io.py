import inspect
import csv
import os
import shutil
import random
import string
from PIL import Image
import numpy as np
import pandas as pd
from distutils.dir_util import copy_tree

'''
1. Root directories, and image extensions of the whole project
'''
sep = os.sep
# root = sep.join(['D:', 'foodquality'])
# root = sep.join(['', 'Users', 'larcuser', 'Google Drive', '_Food Quality Sensing'])
root = sep.join(['', 'Users', 'siyanhu', 'Documents', 'foodquality'])

prepro_dir_list = [root, 'data_cv_preprocess']
original_dir_list = [root, 'data_cv_original']
process_dir_list = [root, 'data_cv_process']
prepro_dir = sep.join(prepro_dir_list)
original_dir = sep.join(original_dir_list)
process_dir = sep.join(process_dir_list)

img_ext_set = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG', 'bmp', 'BMP']
color_set = ['green', 'pattern', 'red']

contour_dir = 'contour'
cropped_dir = 'cropped'
cropped_bg_dir = 'cropped_background'
cropped_mask_dir = 'cropped_mask'

hue_hist_dir = 'hist_hue'
rgb_hist_dir = 'hist_rgb'

objectid_dir = 'objectid'
stat_dir = 'stat'

hsl_dir = 'hsl'
hsl_bg_dir = 'hsl_background'
lum_ref_dir = 'lum_bg'

anno_patches_dir = 'patches'
anno_patches_xml_dir = 'patches_xml'
anno_analysis_dir = 'cropped_anno_patches_analysis'


'''
2. File extension
'''
def get_filename_components(file_path):
    file = ''
    ext = ''
    file_dir = ''
    (file_dir, file) = os.path.split(file_path)
    name_combo = str(file).split('.')
    if len(name_combo) >= 2:
        file = name_combo[0]
        ext = name_combo[-1]
    elif len(name_combo) == 1:
        file = name_combo[0]
    ext = str(ext).replace('.', '')
    return (file_dir, file, ext)


def filter_ext(filepath_list, filter_out_target=False, ext_set=None):
    if ext_set is None:
        return
    unwanted_elems = list()
    for pi_path in filepath_list:
        (file_dir, file, ext) = get_filename_components(pi_path)
        target_exist = False
        if (ext in ext_set) and len(ext) > 0:
            target_exist = True
        if (target_exist == True) and (filter_out_target == True):
            unwanted_elems.append(pi_path)
        elif (target_exist == False) and (filter_out_target == False):
            unwanted_elems.append(pi_path)
    return [ele for ele in filepath_list if ele not in unwanted_elems]


def filter_if_dir(filepath_list, filter_out_target=False):
    #  if isinstance(directory, list):
    rslt = []
    for ff in filepath_list:
        if file_exist(ff) == False:
            continue
        if (os.path.isdir(ff)) and (filter_out_target == False):
            # need to return folder
            rslt.append(ff)
        elif (os.path.isfile(ff)) and (filter_out_target == True):
            # need to return only files
            rslt.append(ff)
    return rslt


def filter_folder(fid_folder_list, filter_out=False, filter_text=''):
    rslt = []
    for ff in fid_folder_list:
        if (filter_text in ff) and (filter_out == False):
            rslt.append(ff)
        elif (not (filter_text in ff)) and (filter_out == True):
            rslt.append(ff)
    return rslt



def replace_file_ext(filepath, new_ext, full_path=True, replace_save=False):
    if (full_path):
        (filedir, file, ext) = get_filename_components(filepath)
        new_file = filedir + sep + file + '.' + new_ext
        if replace_save:
            move_file(filepath, new_file)
        return new_file
    else:
        (file, ext) = str(filepath).split('.')
        new_file = file + '.' + new_ext
        if replace_save:
            move_file(filepath, new_file)
        return new_file


'''
3. File/folder path
'''
def createPath(separator, list_of_dir, file_name=""):
    if len(list_of_dir) <= 0:
        return ""
    while '' in list_of_dir:
        list_of_dir.remove('')
    path_rslt = separator.join(list_of_dir)
    if len(file_name) <= 0:
        return path_rslt
    else:
        return path_rslt + separator + file_name


def traverse_dir(dir, full_path=False, towards_sub=False):
    rslt = list()
    if towards_sub == False:
        file_list = os.listdir(dir)
        for file_name in file_list:
            if full_path:
                rslt.append(os.path.join(dir, file_name))
            else:
                rslt.append(file_name)
        return rslt
    else:
        g = os.walk(dir)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if full_path:
                    rslt.append(os.path.join(path, file_name))
                else:
                    rslt.append(file_name)
        return rslt


def get_nextsub_from_dir_list(dir_list, full_path=False):
    rslt = {}
    for dir in dir_list:
        if '.DS_Store' in dir:
            continue
        sub_list = traverse_dir(dir, full_path=full_path)
        sub_list = filter_ext(sub_list, filter_out_target=True, ext_set='.DS_Store')
        rslt[dir] = sub_list

    return rslt


def file_exist(file_path):
    if 'NA' == file_path:
        return False
    if (file_path == file_path) == False:
        return False
    return os.path.exists(file_path)


def check_file_permission(file_path, destination_path):
    if os.access(file_path, os.R_OK) or os.access(destination_path, os.W_OK):
        return True
    else:
        # print("Permission denied for file movement.", file_path)
        return False

'''
4. File/folder copy, paste, delete and create
'''
def ensure_dir(directory):
    try:
        if isinstance(directory, list):
            directory = createPath(sep, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
    except:
        print("FATAL ERROR: ENSURE_DIR in file_io.py!")


def delete_file(path):
    if os.path.exists(path):  #if file exists
        os.remove(path)
    else:
        print('no such file:%s' % path)


def delete_folder(path):
    if os.path.exists(path):
        if len(os.listdir(path)) == 0:
            os.removedirs(path)
        else:
            try:
                shutil.rmtree(path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
    else:
        print('no such folder:%s' % path)


def move_file(from_dir, to_dir, required_ext=''):
    try:
        if len(required_ext) < 1:
            shutil.move(from_dir, to_dir)
        else:
            cp_func = '*.' + required_ext
            shutil.move(from_dir, to_dir, cp_func)
    except Exception as e:
        print('ERROR: ', e)



def copy_file(from_dir, to_dir, required_ext=''):
    if len(required_ext) < 1:
        shutil.copy(from_dir, to_dir)
    else:
        cp_func = '*.' + required_ext
        shutil.copy(from_dir, to_dir, cp_func)


def copy_folder(from_dir, to_dir):
    copy_tree(from_dir, to_dir)


'''
5. File savings and readings
'''
def save_df_to_csv(rslt_df, file_path, mode='a+', encode='utf_8_sig', breakline='', write_head=True):
    header = list(rslt_df.head())
    with open(file_path, mode, encoding=encode, newline=breakline) as f:
        writer = csv.writer(f)
        if write_head:
            header = list(rslt_df.head())
            writer.writerow(header)
        for index, row in rslt_df.iterrows():
            writer.writerow(row)
        f.close()


def save_dict_to_csv(dict_to_save, file_path, mode='a', encode='utf_8_sig', breakline=''):
    keyword_list = dict_to_save.keys()
    try:
        if not os.path.exists(file_path):
            with open(file_path, "w", newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keyword_list)
                writer.writeheader()

        with open(file_path, mode=mode, newline=breakline, encoding=encode) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keyword_list)
            writer.writerow(dict_to_save)

    except Exception as e:
        print("write error==>", e)
        pass


def save_dict_to_txt(dictionary, filename, mode='w', encode='utf_8_sig'):
    f = open(filename, mode, encoding=encode)
    f.write(str(dictionary))
    f.close()


def read_dict_from_txt(filename, mode='r', encode='utf_8_sig'):
    f = open(filename, mode, encoding=encode)
    a = f.read()
    dictionary = eval(a)
    f.close()
    return dictionary


def save_str_to_txt(str_to_save, file_path, mode='a', encode='utf_8_sig', breakline=''):
    try:
        if not os.path.exists(file_path):
            with open(file_path, "w", newline='', encoding='utf-8') as f:
                f.write(str_to_save)
        else:
            with open(file_path, mode=mode, newline=breakline, encoding=encode) as f:
                f.write(str_to_save)
    except Exception as e:
        print("write error==>", e)
        pass


def save_list_to_csv(list_to_save, file_path, mode='a', encode='utf_8_sig'):
    with open(file_path, mode, encoding=encode, newline='') as f:
        write = csv.writer(f)
        rows = [[data] for data in list_to_save]
        write.writerows(rows)


'''
File Conversions
'''
def image_to_dataframe(image_path):
    frame = inspect.currentframe()
    print('Running func: {} -- {}', inspect.getframeinfo(frame).function, image_path)

    colourImg = Image.open(image_path)
    colourPixels = colourImg.convert("RGB")
    colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
    indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
    allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))
    source_df = pd.DataFrame(allArray, columns=["y", "x", "R", "G", "B"])
    return source_df


'''
Other File Operations
'''
def generate_random_string(rs_len=8):
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(rs_len))
    print("Random string of length", rs_len, "is:", result_str)
    return result_str