# Copyright 2022 The T5Patches Authors.
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

"""Utils for negative and corrective feature converters.

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import seqio
import tensorflow.compat.v2 as tf


def get_first_diff_mask(seq1: tf.Tensor, seq2: tf.Tensor) -> tf.Tensor:
  """Creates one-hot encoded vector that is 1 at the first diff and 0 otherwise.

  Args:
    seq1: 1D tf.Tensor of ints.
    seq2: 1D tf.Tensor of ints. May not be the same length as seq1.

  Returns:
    A 1D tf.Tensor matching the length of seq1.

  """
  output_length = len(seq1)
  pad_size = max(0, output_length - len(seq2))
  padded_or_trimmed_seq2 = tf.pad(seq2[:output_length], [[0, pad_size]])

  def zeros():
    return tf.zeros_like(seq1)

  def first():
    first_diff_idx = tf.argmax(tf.math.not_equal(seq1, padded_or_trimmed_seq2))
    return tf.cast(tf.one_hot(first_diff_idx, output_length), tf.int32)

  return tf.cond(tf.math.reduce_all(seq1 == seq2), zeros, first)


def get_diff_mask(seq1: tf.Tensor, seq2: tf.Tensor) -> tf.Tensor:
  """Creates one-hot encoded vector that is 1 at any diff and 0 otherwise.

  Args:
    seq1: 1D tf.Tensor of ints.
    seq2: 1D tf.Tensor of ints. May not be the same length as seq1.

  Returns:
    A 1D tf.Tensor matching the length of seq1.

  """
  output_length = len(seq1)
  pad_size = max(0, output_length - len(seq2))
  padded_or_trimmed_seq2 = tf.pad(seq2[:output_length], [[0, pad_size]])
  return seqio.feature_converters.non_padding_position(seq1) * (
      tf.cast(tf.math.not_equal(seq1, padded_or_trimmed_seq2), tf.int32))
