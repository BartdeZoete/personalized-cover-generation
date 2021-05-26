##### Copyright 2021 Google LLC.

# Licensed under the Apache License, Version 2.0 (the "License");


# the code was altered to be usable outside of Google Colaboratory and also
# allow for fully automatic processing


# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")
import os
import time
import crepe
import ddsp
import ddsp.training
from utils import (
    detect_notes, fit_quantile_transform,
    DEFAULT_SAMPLE_RATE, to_file,
    QuantileTransformer,
    shift_f0, shift_ld)
import gin
import numpy as np
import pickle
import tensorflow as tf
tf.autograph.set_verbosity(0)
import note_seq


def process(curr_dir, path, fname, model_name, load_preprocessed=False):
    model_name = model_name.lower()

    sample_rate = DEFAULT_SAMPLE_RATE  # 16000

    audio = open(f"{path}{fname}", 'rb').read()
    audio = note_seq.audio_io.wav_data_to_samples_pydub(wav_data=audio, sample_rate=DEFAULT_SAMPLE_RATE, normalize_db=None)

    audio = audio[np.newaxis, :]
    audio_shape1 = audio.shape[1]

    # Setup the session.
    ddsp.spectral_ops.reset_crepe()

    # Compute features.
    if not load_preprocessed:
        print("Computing features... can take quite long")
        audio_features = ddsp.training.metrics.compute_audio_features(audio)
        audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)

        np.save(f"temp/{model_name}_audio.npy", audio_features['audio'])
        np.save(f"temp/{model_name}_loudness_db.npy", audio_features['loudness_db'])
        np.save(f"temp/{model_name}_f0_hz.npy", audio_features['f0_hz'])
        np.save(f"temp/{model_name}_f0_confidence.npy", audio_features['f0_confidence'])
    else:
        print("Loading features from files")

    a = np.load(f"temp/{model_name}_audio.npy")
    b = np.load(f"temp/{model_name}_loudness_db.npy")
    c = np.load(f"temp/{model_name}_f0_hz.npy")
    d = np.load(f"temp/{model_name}_f0_confidence.npy")
    audio_features = {'audio':a, 'loudness_db':b, 'f0_hz':c, 'f0_confidence':d}
    a, b, c, d = None, None, None, None
    audio = None


    if "bass" in fname or "drum" in fname:
        threshold = 0.2 # 1 by default, 0.2 needed for bass tracks
    else:
        threshold = 1.0


    TRIM = -15

    if model_name in ('violin', 'flute', 'flute2', 'trumpet', 'tenor_saxophone'):
      # Pretrained models.
      PRETRAINED_DIR = curr_dir + 'pretrained'
      model_dir = PRETRAINED_DIR
      gin_file = os.path.join(model_dir, f'{model_name}_operative_config-0.gin')


    # Parse gin config,
    with gin.unlock_config():
      gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if f'{model_name}_ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio_shape1 / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
      gin.parse_config(gin_params)


    # Trim all input vectors to correct lengths
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
      audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:, :n_samples]


    # Set up the model just to predict audio given new conditioning
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)
    ckpt = None

    print("Restoring model")
    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)
    print('Restoring model took %.1f seconds' % (time.time() - start_time))


    quiet = 20
    pitch_shift =  0
    loudness_shift = 0

    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}

    # Detect sections that are "on".
    mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                        audio_features['f0_confidence'],
                                        threshold)

    # Load the dataset statistics.
    if model_name == 'flute2':
        mean_pitch = 73.08097
    elif model_name == "trumpet":
        mean_pitch = 68.572334
    elif model_name == "violin":
        mean_pitch = 72.06872
    elif model_name == "flute":
        mean_pitch = 76.75298
    elif model_name == "tenor_saxophone":
        mean_pitch = 59.01674
    DATASET_STATS = {"mean_pitch" : mean_pitch}
    dataset_stats_file = os.path.join(model_dir, f'{model_name}.qtr')
    print(f'Loading dataset statistics from {dataset_stats_file}')
    try:
        with open(dataset_stats_file, 'rb') as f:
            DATASET_STATS['quantile_transform'] = pickle.load(f)
    except Exception as err:
      print('\nLoading dataset statistics from pickle failed: {}.'.format(err))
      raise

    print("Loading success")

    # Shift the pitch register.
    target_mean_pitch = DATASET_STATS['mean_pitch']
    pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
    mean_pitch = np.mean(pitch[mask_on])
    p_diff = target_mean_pitch - mean_pitch
    p_diff_octave = p_diff / 12.0
    round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
    p_diff_octave = round_fn(p_diff_octave)
    audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)


    # Quantile shift the note_on parts.
    _, loudness_norm = fit_quantile_transform(
        audio_features['loudness_db'],
        mask_on,
        inv_quantile=DATASET_STATS['quantile_transform'])

    # Turn down the note_off parts.
    mask_off = np.logical_not(mask_on)
    loudness_norm[mask_off] -=  quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
    loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)

    audio_features_mod['loudness_db'] = loudness_norm

    # Resynthesize Audio
    af = audio_features if audio_features_mod is None else audio_features_mod

    # Run a batch of predictions.
    outputs = model(af, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)

    out_file = f"{path}{fname.split('.')[0]}_{model_name}.wav"
    return audio_gen, out_file, sample_rate
