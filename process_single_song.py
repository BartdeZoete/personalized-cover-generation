import translation_pipeline as tp

# directory from root where the current file is located
curr_dir = ""
# path to the song to translate, relative to the current dir
# output song will be placed in the same dir
fname = ""

from_genre = "pop"
to_genre = "jazz"

tp.translate(curr_dir, curr_dir, fname, from_genre, to_genre)
