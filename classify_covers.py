from cover_detector_binary import binary_cover_classification
import translation_pipeline as tp
from utils import QuantileTransformer
import numpy as np

#information about where to find files
# path to the songs to process
path = ""
# absolute path to the current directory
curr_dir = ""
# file that stores the names of the songs to process, by default assumed to be
# located in `path`
file_read = 'downloadsNames.txt'


#open the files to convert
correct_classifications = []
f = open(path+file_read, 'r')
for row in f:
    if row.strip() == '':
        continue

    info = row.split(',')
    mp3_file_name = info[0]
    cluster_number = int(info[1].strip('\n'))

    #tp.translate(curr_dir, path, mp3_file_name, "pop", "jazz")
    wav_file_name = mp3_file_name[:-4]+'_jazz.wav'

    is_cover = binary_cover_classification(path, wav_file_name, cluster_number)
    correct_classifications.append(is_cover)

total = len(correct_classifications)
correct = np.sum(correct_classifications)
print("Accuracy:", correct/total)
