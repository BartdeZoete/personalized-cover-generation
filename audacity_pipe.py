import librosa
import os
import sys


if sys.platform == 'win32':
    print("pipe-test.py, running on windows")
    TONAME = '\\\\.\\pipe\\ToSrvPipe'
    FROMNAME = '\\\\.\\pipe\\FromSrvPipe'
    EOL = '\r\n\0'
else:
    print("pipe-test.py, running on linux or mac")
    TONAME = '/tmp/audacity_script_pipe.to.' + str(os.getuid())
    FROMNAME = '/tmp/audacity_script_pipe.from.' + str(os.getuid())
    EOL = '\n'

print("Write to  \"" + TONAME +"\"")
if not os.path.exists(TONAME):
    print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()

print("Read from \"" + FROMNAME +"\"")
if not os.path.exists(FROMNAME):
    print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()

print("-- Both pipes exist.  Good.")

TOFILE = open(TONAME, 'w')
print("-- File to write to has been opened")
FROMFILE = open(FROMNAME, 'rt')
print("-- File to read from has now been opened too\r\n")


def send_command(command):
    """Send a single command."""
    print("Send: >>> \n"+command)
    TOFILE.write(command + EOL)
    TOFILE.flush()

def get_response():
    """Return the command response."""
    result = ''
    line = ''
    while True:
        result += line
        line = FROMFILE.readline()
        if line == '\n' and len(result) > 0:
            break
    return result

def do_command(command):
    """Send one command, and return the response."""
    send_command(command)
    response = get_response()
    print("Rcvd: <<< \n" + response)
    return response

def import_as_track(path, track_id):
    do_command(f"Select: Track={track_id}")
    do_command(f"Import2: Filename={path}")

def adjust_bass_and_treble(bass_db_shift, treble_db_shift):
    do_command(f"BassAndTreble: Bass={bass_db_shift}, Treble={treble_db_shift}")

def tempo(amount):
    do_command(f"ChangeTempo: Percentage={amount}")

def select_all():
    do_command("SelectAll:")

def select_track_id(track_id):
    select_all()
    do_command(f'SelectTracks:Mode="Set" Track="{track_id}" TrackCount="1"')

def deselect():
    do_command("SelectNone:")

def amplify(amount):
    do_command(f"SetTrackAudio: Gain={amount}")

def export(fname):
    select_all()
    do_command(f"Export2: Filename={fname}")

# perform the pop to jazz translation
def pop_to_jazz(tracks_path, bpm, out_file):
    import_as_track(tracks_path+"vocals.wav", 0)
    import_as_track(tracks_path+"drums.wav", 1)
    import_as_track(tracks_path+"other.wav", 2)
    import_as_track(tracks_path+"bass.wav", 3)
    import_as_track(tracks_path+"othermono_flute2.wav", 4)
    import_as_track(tracks_path+"bassmono_trumpet.wav", 5)

    # adjust bpm for all tracks
    select_all()
    new_bpm = bpm - 15
    tempo((new_bpm-bpm) / bpm * 100.0)
    deselect()

    # vocals
    select_track_id(0)
    amplify(-2)
    deselect()

    # drums
    select_track_id(1)
    adjust_bass_and_treble(-12, 6)
    amplify(-13)
    deselect()

    # other
    select_track_id(2)
    amplify(-15)
    deselect()

    # bass
    select_track_id(3)
    amplify(-20)
    deselect()

    # other flute2
    select_track_id(4)
    amplify(-6)
    deselect()

    # bass trumpet
    select_track_id(5)
    amplify(6)
    deselect()

    export(out_file)
    select_all()
    do_command("RemoveTracks:")
    do_command("SelectNone:")


def process(original_file, tracks_path, from_genre, to_genre):
    y, sr = librosa.load(original_file, duration=30)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    fname = original_file.split('/')[-1]
    path = original_file[:-len(fname)]
    out_file = f"{path}{fname.split('.')[0]}_{to_genre}.wav"
    print(out_file)
    if from_genre == "pop" and to_genre == "jazz":
        pop_to_jazz(tracks_path, bpm, out_file)
