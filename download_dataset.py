import sys
import urllib
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join

IMAGE_FOLDER = "data/"

downloaded_files = [f for f in listdir(IMAGE_FOLDER) if isfile(join(IMAGE_FOLDER, f))]
files_set = set(downloaded_files)


def download_image(line):
    try:
        image_id = line.split("\t")[0]
        image_url = line.split("\t")[1]
        image_extension = image_url.split(".")[-1]
        image_extension = image_extension.replace("\n", "")
        file_name = image_id + "." + image_extension
        file_path = IMAGE_FOLDER + file_name
        if file_name not in files_set:
            urllib.urlretrieve(image_url, file_path)
        files_set.add(file_name)
    except Exception as e:
        print(e)
        return


url_file = sys.argv[1]
process_pool = Pool(10)


def get_images_urls():
    with open(url_file, 'r') as file_stream:
        line = file_stream.readline()
        while line:
            line = file_stream.readline()
            if len(line) == 0:
                continue
            yield line


process_pool.map(download_image, get_images_urls())
