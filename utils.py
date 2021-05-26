# Copyright 2021 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# this code was copied and altered as to be usable outside of Google
# Colaboratory
# the get_model function is not from the DDSP library


import ddsp
import numpy as np
from scipy import stats
import tensorflow as tf
tf.autograph.set_verbosity(0)
from scipy.io import wavfile
import librosa
import io

DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE

_play_count = 0  # Used for ephemeral play().

def get_model(fname, from_genre, to_genre):
    if from_genre == "pop" and to_genre == "jazz":
        if fname.startswith("bass"):
            return "trumpet"
        elif fname.startswith("other"):
            return "flute2"
        else:
            return None
    print(f"Translations from {from_genre} to {to_genre} are unsupported")
    return None

def shift_ld(audio_features, ld_shift=0.0):
  """Shift loudness by a number of ocatves."""
  audio_features['loudness_db'] += ld_shift
  return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
  """Shift f0 by a number of ocatves."""
  audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
  audio_features['f0_hz'] = np.clip(audio_features['f0_hz'],
                                    0.0,
                                    librosa.midi_to_hz(110.0))
  return audio_features

def to_file(array_of_floats,
         fname,
         sample_rate=DEFAULT_SAMPLE_RATE,
         ephemeral=True,
         autoplay=False):
  # If batched, take first element.
  if len(array_of_floats.shape) == 2:
    array_of_floats = array_of_floats[0]

  normalizer = float(np.iinfo(np.int16).max)
  array_of_ints = np.array(
      np.asarray(array_of_floats) * normalizer, dtype=np.int16)
  memfile = io.BytesIO()
  wavfile.write(memfile, sample_rate, array_of_ints)
  wavfile.write(fname, sample_rate, array_of_ints)


# ------------------------------------------------------------------------------
# Loudness Normalization
# ------------------------------------------------------------------------------
def smooth(x, filter_size=3):
  """Smooth 1-d signal with a box filter."""
  x = tf.convert_to_tensor(x, tf.float32)
  is_2d = len(x.shape) == 2
  x = x[:, :, tf.newaxis] if is_2d else x[tf.newaxis, :, tf.newaxis]
  w = tf.ones([filter_size])[:, tf.newaxis, tf.newaxis] / float(filter_size)
  y = tf.nn.conv1d(x, w, stride=1, padding='SAME')
  y = y[:, :, 0] if is_2d else y[0, :, 0]
  return y.numpy()


def detect_notes(loudness_db,
                 f0_confidence,
                 note_threshold=1.0,
                 exponent=2.0,
                 smoothing=40,
                 f0_confidence_threshold=0.7,
                 min_db=-120.):
  """Detect note on-off using loudness and smoothed f0_confidence."""
  mean_db = np.mean(loudness_db)
  db = smooth(f0_confidence**exponent, smoothing) * (loudness_db - min_db)
  db_threshold = (mean_db - min_db) * f0_confidence_threshold**exponent
  note_on_ratio = db / db_threshold
  mask_on = note_on_ratio >= note_threshold
  return mask_on, note_on_ratio


class QuantileTransformer:
  """Transform features using quantiles information.

  Stripped down version of sklearn.preprocessing.QuantileTransformer.
  https://github.com/scikit-learn/scikit-learn/blob/
  863e58fcd5ce960b4af60362b44d4f33f08c0f97/sklearn/preprocessing/_data.py

  Putting directly in ddsp library to avoid dependency on sklearn that breaks
  when pickling and unpickling from different versions of sklearn.
  """

  def __init__(self,
               n_quantiles=1000,
               output_distribution='uniform',
               subsample=int(1e5)):
    """Constructor.

    Args:
      n_quantiles: int, default=1000 or n_samples Number of quantiles to be
        computed. It corresponds to the number of landmarks used to discretize
        the cumulative distribution function. If n_quantiles is larger than the
        number of samples, n_quantiles is set to the number of samples as a
        larger number of quantiles does not give a better approximation of the
        cumulative distribution function estimator.
      output_distribution: {'uniform', 'normal'}, default='uniform' Marginal
        distribution for the transformed data. The choices are 'uniform'
        (default) or 'normal'.
      subsample: int, default=1e5 Maximum number of samples used to estimate
        the quantiles for computational efficiency. Note that the subsampling
        procedure may differ for value-identical sparse and dense matrices.
    """
    self.n_quantiles = n_quantiles
    self.output_distribution = output_distribution
    self.subsample = subsample
    self.random_state = np.random.mtrand._rand

  def _dense_fit(self, x, random_state):
    """Compute percentiles for dense matrices.

    Args:
      x: ndarray of shape (n_samples, n_features)
        The data used to scale along the features axis.
      random_state: Numpy random number generator.
    """
    n_samples, _ = x.shape
    references = self.references_ * 100

    self.quantiles_ = []
    for col in x.T:
      if self.subsample < n_samples:
        subsample_idx = random_state.choice(
            n_samples, size=self.subsample, replace=False)
        col = col.take(subsample_idx, mode='clip')
      self.quantiles_.append(np.nanpercentile(col, references))
    self.quantiles_ = np.transpose(self.quantiles_)
    # Due to floating-point precision error in `np.nanpercentile`,
    # make sure that quantiles are monotonically increasing.
    # Upstream issue in numpy:
    # https://github.com/numpy/numpy/issues/14685
    self.quantiles_ = np.maximum.accumulate(self.quantiles_)

  def fit(self, x):
    """Compute the quantiles used for transforming.

    Parameters
    ----------
    Args:
      x: {array-like, sparse matrix} of shape (n_samples, n_features)
        The data used to scale along the features axis. If a sparse
        matrix is provided, it will be converted into a sparse
        ``csc_matrix``. Additionally, the sparse matrix needs to be
        nonnegative if `ignore_implicit_zeros` is False.

    Returns:
      self: object
         Fitted transformer.
    """
    if self.n_quantiles <= 0:
      raise ValueError("Invalid value for 'n_quantiles': %d. "
                       'The number of quantiles must be at least one.' %
                       self.n_quantiles)
    n_samples = x.shape[0]
    self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

    # Create the quantiles of reference
    self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
    self._dense_fit(x, self.random_state)
    return self

  def _transform_col(self, x_col, quantiles, inverse):
    """Private function to transform a single feature."""
    output_distribution = self.output_distribution
    bounds_threshold = 1e-7

    if not inverse:
      lower_bound_x = quantiles[0]
      upper_bound_x = quantiles[-1]
      lower_bound_y = 0
      upper_bound_y = 1
    else:
      lower_bound_x = 0
      upper_bound_x = 1
      lower_bound_y = quantiles[0]
      upper_bound_y = quantiles[-1]
      # for inverse transform, match a uniform distribution
      with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
        if output_distribution == 'normal':
          x_col = stats.norm.cdf(x_col)
        # else output distribution is already a uniform distribution

    # find index for lower and higher bounds
    with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
      if output_distribution == 'normal':
        lower_bounds_idx = (x_col - bounds_threshold < lower_bound_x)
        upper_bounds_idx = (x_col + bounds_threshold > upper_bound_x)
      if output_distribution == 'uniform':
        lower_bounds_idx = (x_col == lower_bound_x)
        upper_bounds_idx = (x_col == upper_bound_x)

    isfinite_mask = ~np.isnan(x_col)
    x_col_finite = x_col[isfinite_mask]
    if not inverse:
      # Interpolate in one direction and in the other and take the
      # mean. This is in case of repeated values in the features
      # and hence repeated quantiles
      #
      # If we don't do this, only one extreme of the duplicated is
      # used (the upper when we do ascending, and the
      # lower for descending). We take the mean of these two
      x_col[isfinite_mask] = .5 * (
          np.interp(x_col_finite, quantiles, self.references_) -
          np.interp(-x_col_finite, -quantiles[::-1], -self.references_[::-1]))
    else:
      x_col[isfinite_mask] = np.interp(x_col_finite, self.references_,
                                       quantiles)

    x_col[upper_bounds_idx] = upper_bound_y
    x_col[lower_bounds_idx] = lower_bound_y
    # for forward transform, match the output distribution
    if not inverse:
      with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
        if output_distribution == 'normal':
          x_col = stats.norm.ppf(x_col)
          # find the value to clip the data to avoid mapping to
          # infinity. Clip such that the inverse transform will be
          # consistent
          clip_min = stats.norm.ppf(bounds_threshold - np.spacing(1))
          clip_max = stats.norm.ppf(1 - (bounds_threshold - np.spacing(1)))
          x_col = np.clip(x_col, clip_min, clip_max)
        # else output distribution is uniform and the ppf is the
        # identity function so we let x_col unchanged

    return x_col

  def _transform(self, x, inverse=False):
    """Forward and inverse transform.

    Args:
      x : ndarray of shape (n_samples, n_features)
        The data used to scale along the features axis.
      inverse : bool, default=False
        If False, apply forward transform. If True, apply
        inverse transform.

    Returns:
      x : ndarray of shape (n_samples, n_features)
        Projected data
    """
    x = np.array(x)  # Explicit copy.
    for feature_idx in range(x.shape[1]):
      x[:, feature_idx] = self._transform_col(
          x[:, feature_idx], self.quantiles_[:, feature_idx], inverse)
    return x

  def transform(self, x):
    """Feature-wise transformation of the data."""
    return self._transform(x, inverse=False)

  def inverse_transform(self, x):
    """Back-projection to the original space."""
    return self._transform(x, inverse=True)

  def fit_transform(self, x):
    """Fit and transform."""
    return self.fit(x).transform(x)


def fit_quantile_transform(loudness_db, mask_on, inv_quantile=None):
  """Fits quantile normalization, given a note_on mask.

  Optionally, performs the inverse transformation given a pre-fitted transform.
  Args:
    loudness_db: Decibels, shape [batch, time]
    mask_on: A binary mask for when a note is present, shape [batch, time].
    inv_quantile: Optional pretrained QuantileTransformer to perform the inverse
      transformation.

  Returns:
    Trained quantile transform. Also returns the renormalized loudnesses if
      inv_quantile is provided.
  """
  quantile_transform = QuantileTransformer()
  loudness_flat = np.ravel(loudness_db[mask_on])[:, np.newaxis]
  loudness_flat_q = quantile_transform.fit_transform(loudness_flat)

  if inv_quantile is None:
    return quantile_transform
  else:
    loudness_flat_norm = inv_quantile.inverse_transform(loudness_flat_q)
    loudness_norm = np.ravel(loudness_db.copy())[:, np.newaxis]
    loudness_norm[mask_on] = loudness_flat_norm
    return quantile_transform, loudness_norm
