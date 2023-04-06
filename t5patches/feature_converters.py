# Copyright 2023 The T5Patches Authors.
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

"""feature converters for corrections.

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
import abc
from typing import Mapping

import gin
import seqio
from t5patches import feature_converters_utils
import tensorflow.compat.v2 as tf


class NegativeTrainingFeatureConverterBase(
    seqio.feature_converters.FeatureConverter, abc.ABC):
  """Base feature converter for negative or corrective training.

  The input dataset must have "inputs," "negative_targets,"
  and "corrected_targets" field.

  Most classes which inherit from this one differ primarily in which targets
  are used for decoder_target_tokens, as well as how decoder_loss_weights is
  computed.

  """

  TASK_FEATURES = {
      "inputs":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "negative_targets":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "corrected_targets":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }

  MODEL_FEATURES = {
      "encoder_input_tokens":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_tokens":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "decoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
      "decoder_positions": tf.int32,
  }

  @abc.abstractmethod
  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    """Calculates decoder_loss_weights for an example.

    Uses values from "negative_targets" and "corrected_targets" from
    `features` to get the weights associated with each decoder output token for
    calculating the loss.

    Args:
      features: A dict mapping str key names to their tf.Tensor values. Notably
        expects "negative_targets" and "corrected_targets" as keys.  Returns a
        1D tf.Tensor of weights.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    """Specifies decoder_target_tokens for an example.

    This function determines whether "negative_targets" or "corrected_targets"
    should used as the decoder target on a per-example basis given `features`.

    Args:
      features: A dict mapping str key names to their tf.Tensor values. Notably
        expects "negative_targets" and "corrected_targets" as keys.  Returns a
        1D tf.Tensor of token ids.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    """Determines the sequence length for decoder outputs.

    This function determines whether the lengths specified for
    "negative_targets" or "corrected_targets" should used as the decoder length.

    Args:
      task_feature_lengths: A dict mapping str key names to their lengths.
        Notably expects "negative_targets" and "corrected_targets" as keys.
        Returns an int.
    """
    raise NotImplementedError

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      # targets_segment_id is present only for a packed dataset.
      decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
          self.get_decoder_target_tokens(features),
          sequence_id=features.get("targets_segment_ids", None))

      d = {
          "encoder_input_tokens": features["inputs"],
          "decoder_target_tokens": self.get_decoder_target_tokens(features),
          "decoder_input_tokens": decoder_input_tokens,
          "decoder_loss_weights": self.get_decoder_weights(features),
      }

      if self.pack:
        raise NotImplementedError(
            "Packing currently not supported with negative training featurizers."
        )

      return d

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = self.get_decoder_length(task_feature_lengths)

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
    }
    if self.pack:
      model_feature_lengths["encoder_segment_ids"] = encoder_length
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["encoder_positions"] = encoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths


@gin.configurable(module="feature_converters")
class NegativeTrainingFirstFeatureConverter(NegativeTrainingFeatureConverterBase
                                           ):
  """Feature converter for negative training.

  The input dataset must have "inputs," "negative_targets,"
  and "corrected_targets" field.

  The decoder_target_tokens will be negative_targets.

  For decoder_loss_weights, only one token will be marked with a negative
  weight, i.e. the first occurrence of a token diff between negative_targets
  and corrected_targets. All other tokens will have weights of 1.
  Padding tokens will have weights of 0.

  Example:
  ds = [
    {
      "inputs": [7, 8, 5, 1],
      "negative_targets": [3, 9, 1],
      "corrected_targets": [3, 8, 2]
    },
    {
      "inputs": [8, 4, 9, 3, 1],
      "negative_targets": [4, 1],
      "corrected_targets": [4, 2]
    }]

  task_feature_lengths = {"inputs": 5, "targets": 3}

  converted_ds = [{
       "encoder_input_tokens": [7, 8, 5, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 0],
          "encoder_positions": [0, 1, 2, 3, 0],
      "decoder_target_tokens": [3, 9, 1],
       "decoder_input_tokens": [0, 3, 9],
       "decoder_loss_weights": [1, -1, 1],
        "decoder_segment_ids": [1, 1, 1],
          "decoder_positions": [0, 1, 2],
  },
  {
       "encoder_input_tokens": [8, 4, 9, 3, 1],
        "encoder_segment_ids": [1, 1, 1, 1, 1],
          "encoder_positions": [0, 1, 2, 3, 4],
      "decoder_target_tokens": [4, 1, 0],
       "decoder_input_tokens": [0, 4, 0],
       "decoder_loss_weights": [1, -1, 1],
        "decoder_segment_ids": [1, 1, 0],
          "decoder_positions": [0, 1, 0],
  }]
  """

  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return feature_converters_utils.get_first_diff_mask(
        features["negative_targets"], features["corrected_targets"]
    ) * -2 + seqio.feature_converters.non_padding_position(
        features["negative_targets"]
    )

  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return features["negative_targets"]

  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    return task_feature_lengths["negative_targets"]


@gin.configurable(module="feature_converters")
class NegativeTrainingDiffFeatureConverter(NegativeTrainingFeatureConverterBase
                                          ):
  """Feature converter for negative training.

  The input dataset must have "inputs," "negative_targets,"
  and "corrected_targets" field.

  The decoder_target_tokens will be negative_targets.

  All tokens from negative_targets that differ from the corresponding
  corrected_targets will be marked with a negative weight.
  All other tokens will have weights of 1.
  Padding tokens will have weights of 0.

  Example:
  ds = [
    {
      "inputs": [7, 8, 5, 1],
      "negative_targets": [3, 9, 1],
      "corrected_targets": [3, 8, 2]
    },
    {
      "inputs": [8, 4, 9, 3, 1],
      "negative_targets": [4, 1],
      "corrected_targets": [4, 2]
    }]

  task_feature_lengths = {"inputs": 5, "targets": 3}

  converted_ds = [{
       "encoder_input_tokens": [7, 8, 5, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 0],
          "encoder_positions": [0, 1, 2, 3, 0],
      "decoder_target_tokens": [3, 9, 1],
       "decoder_input_tokens": [0, 3, 9],
       "decoder_loss_weights": [1, -1, -1],
        "decoder_segment_ids": [1, 1, 1],
          "decoder_positions": [0, 1, 2],
  },
  {
       "encoder_input_tokens": [8, 4, 9, 3, 1],
        "encoder_segment_ids": [1, 1, 1, 1, 1],
          "encoder_positions": [0, 1, 2, 3, 4],
      "decoder_target_tokens": [4, 1, 0],
       "decoder_input_tokens": [0, 4, 0],
       "decoder_loss_weights": [1, -1, 1],
        "decoder_segment_ids": [1, 1, 0],
          "decoder_positions": [0, 1, 0],
  }]
  """

  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return feature_converters_utils.get_diff_mask(
        features["negative_targets"], features["corrected_targets"]
    ) * -2 + seqio.feature_converters.non_padding_position(
        features["negative_targets"]
    )

  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return features["negative_targets"]

  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    return task_feature_lengths["negative_targets"]


@gin.configurable(module="feature_converters")
class NegativeTrainingFullFeatureConverter(NegativeTrainingFeatureConverterBase
                                          ):
  """Feature converter for negative training.

  All tokens from negative_targets will be marked with a negative weight of -1.

  Example:
  ds = [
    {
      "inputs": [7, 8, 5, 1],
      "negative_targets": [3, 9, 1],
      "corrected_targets": [3, 8, 2]
    },
    {
      "inputs": [8, 4, 9, 3, 1],
      "negative_targets": [4, 1],
      "corrected_targets": [4, 2]
    }]

  task_feature_lengths = {"inputs": 5, "targets": 3}

  converted_ds = [{
       "encoder_input_tokens": [7, 8, 5, 1, 0],
        "encoder_segment_ids": [1, 1, 1, 1, 0],
          "encoder_positions": [0, 1, 2, 3, 0],
      "decoder_target_tokens": [3, 9, 1],
       "decoder_input_tokens": [0, 3, 9],
       "decoder_loss_weights": [-1, -1, -1],
        "decoder_segment_ids": [1, 1, 1],
          "decoder_positions": [0, 1, 2],
  },
  {
       "encoder_input_tokens": [8, 4, 9, 3, 1],
        "encoder_segment_ids": [1, 1, 1, 1, 1],
          "encoder_positions": [0, 1, 2, 3, 4],
      "decoder_target_tokens": [4, 1, 0],
       "decoder_input_tokens": [0, 4, 0],
       "decoder_loss_weights": [-1, -1, 0],
        "decoder_segment_ids": [1, 1, 0],
          "decoder_positions": [0, 1, 0],
  }]

  """

  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return tf.cast(
        seqio.feature_converters.non_padding_position(
            features["negative_targets"]) * -1, tf.int32)

  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return features["negative_targets"]

  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    return task_feature_lengths["negative_targets"]


@gin.configurable(module="feature_converters")
class CorrectiveTrainingDiffFeatureConverter(
    NegativeTrainingFeatureConverterBase):
  """Feature converter using corrections only.

  The input dataset must have "inputs," "negative_targets,"
  and "corrected_targets" field. The behavior is the same as
  NegativeTrainingDiffFeatureConverter except for the decoder_target_tokens,
  which use corrected_targets instead of negative_targets.

  The decoder_loss_weights are 1 where corrected_targets differs from
  negative_targets and 0 otherwise.

  task_feature_lengths = {"inputs": 5, "targets": 4}

  Example:
  ds = [
    {
      "inputs": [7, 8, 5, 1],
      "negative_targets": [3, 8, 1],
      "corrected_targets": [3, 9, 2]
    }]

  converted_ds = [{
      ...
      "decoder_target_tokens": [3, 9, 2, 0],
       "decoder_input_tokens": [0, 3, 9, 0],
       "decoder_loss_weights": [0, 1, 1, 0],
        "decoder_segment_ids": [1, 1, 1, 0],
          "decoder_positions": [0, 1, 2, 0],
  }]

  """

  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return feature_converters_utils.get_diff_mask(features["corrected_targets"],
                                                  features["negative_targets"])

  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return features["corrected_targets"]

  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    return task_feature_lengths["corrected_targets"]


@gin.configurable(module="feature_converters")
class CorrectiveTrainingFirstFeatureConverter(
    NegativeTrainingFeatureConverterBase):
  """Feature converter for "negative training" using positive corrections only.

  The input dataset must have "inputs," "negative_targets,"
  and "corrected_targets" field. The behavior is the same as
  NegativeTrainingFirstFeatureConverter except for the decoder_target_tokens,
  which use corrected_targets instead of negative_targets.

  The decoder_loss_weights are 1 at the first index where corrected_targets
  differs from negative_targets and 0 otherwise.

  Example:
  ds = [
    {
      "inputs": [7, 8, 5, 1],
      "negative_targets": [3, 8, 1],
      "corrected_targets": [3, 9, 2]
    }]

  task_feature_lengths = {"inputs": 5, "targets": 4}

  converted_ds = [{
      ...
      "decoder_target_tokens": [3, 9, 2, 0],
       "decoder_input_tokens": [0, 3, 9, 0],
       "decoder_loss_weights": [0, 1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 0],
          "decoder_positions": [0, 1, 2, 0],
  }]

  """

  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return feature_converters_utils.get_first_diff_mask(
        features["corrected_targets"], features["negative_targets"])

  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return features["corrected_targets"]

  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    return task_feature_lengths["corrected_targets"]


@gin.configurable(module="feature_converters")
class CorrectiveTrainingFullFeatureConverter(
    NegativeTrainingFeatureConverterBase):
  """Feature converter for "negative training" using positive corrections only.

  The input dataset must have "inputs," "negative_targets,"
  and "corrected_targets" field. The behavior is the same as
  NegativeTrainingFullFeatureConverter except for the decoder outputs.

  Here, decoder_loss_weights are 1 for all non-padded tokens. This is equivalent
  to regular training with corrected_targets as output.

  Example:
  ds = [
    {
      "inputs": [7, 8, 5, 1],
      "negative_targets": [3, 8, 1],
      "corrected_targets": [3, 9, 2]
    }]

  converted_ds = [{
      ...
      "decoder_target_tokens": [3, 9, 2, 0],
       "decoder_input_tokens": [0, 3, 9, 0],
       "decoder_loss_weights": [1, 1, 1, 0],
        "decoder_segment_ids": [1, 1, 1, 0],
          "decoder_positions": [0, 1, 2, 0],
  }]

  """

  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return seqio.feature_converters.non_padding_position(
        features["corrected_targets"])

  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    return features["corrected_targets"]

  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    return task_feature_lengths["corrected_targets"]


@gin.configurable(module="feature_converters")
class NegativeAndPositiveTrainingFeatureConverterBase(
    NegativeTrainingFeatureConverterBase, abc.ABC):
  """Feature converter for negative and positive examples.

  We distinguish positive examples from corrections. The latter is a good
  example that corrects for a negative example, whereas the former is a good
  example without a corresponding negative example.

  Feature converters based on this base assume the input dataset have "inputs,"
  "negative_targets," and "corrected_targets" fields, where a positive example
  gets an output sequence in the corrected_targets field and a dummy sequence
  in the negative_targets field, while a negative example gets an output
  sequence in the negative_targets field and a dummy sequence in the
  negative_targets field. A dummy sequence for positive examples should be
  shorter than the length of corrected_targets. A dummy sequence of negative
  examples should have length less than or equal to the length of
  negative_targets.

  """
  NEGATIVE_FEATURE_CONVERTER_CLS = NegativeTrainingFullFeatureConverter
  POSITIVE_FEATURE_CONVERTER_CLS = CorrectiveTrainingFullFeatureConverter

  def __init__(self,
               pack: bool = False,
               use_custom_packing_ops: bool = False,
               apply_length_check: bool = True):
    if pack:
      raise NotImplementedError(
          "Packing currently not supported with negative training featurizers.")
    super().__init__(pack, use_custom_packing_ops, apply_length_check)
    self.negative_feature_converter = self.NEGATIVE_FEATURE_CONVERTER_CLS(
        pack=self._pack,
        use_custom_packing_ops=self._use_custom_packing_ops,
        apply_length_check=self._apply_length_check)
    self.positive_feature_converter = self.POSITIVE_FEATURE_CONVERTER_CLS(
        pack=self._pack,
        use_custom_packing_ops=self._use_custom_packing_ops,
        apply_length_check=self._apply_length_check)

  def _get_active_output_lengths(self, features):
    negative_output_length = tf.math.reduce_sum(
        seqio.feature_converters.non_padding_position(
            features["negative_targets"]))
    positive_output_length = tf.math.reduce_sum(
        seqio.feature_converters.non_padding_position(
            features["corrected_targets"]))
    return negative_output_length, positive_output_length

  def get_decoder_weights(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    negative_output_length, positive_output_length = (
        self._get_active_output_lengths(features))
    if negative_output_length >= positive_output_length:
      return self.negative_feature_converter.get_decoder_weights(features)
    else:
      return self.positive_feature_converter.get_decoder_weights(features)

  def get_decoder_target_tokens(self,
                                features: Mapping[str, tf.Tensor]) -> tf.Tensor:
    negative_output_length, positive_output_length = (
        self._get_active_output_lengths(features))
    if negative_output_length >= positive_output_length:
      return self.negative_feature_converter.get_decoder_target_tokens(features)
    else:
      return self.positive_feature_converter.get_decoder_target_tokens(features)

  def get_decoder_length(self, task_feature_lengths: Mapping[str, int]) -> int:
    neg_len = self.negative_feature_converter.get_decoder_length(
        task_feature_lengths)
    pos_len = self.positive_feature_converter.get_decoder_length(
        task_feature_lengths)
    return max(neg_len, pos_len)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    negative_decoder_length = (
        self.negative_feature_converter.get_decoder_length(task_feature_lengths)
    )
    try:
      positive_decoder_length = (
          self.positive_feature_converter.get_decoder_length(
              task_feature_lengths))
    except AttributeError:
      positive_decoder_length = negative_decoder_length
    if positive_decoder_length != negative_decoder_length:
      raise ValueError("""
      Negative and corrected targets should be assigned the same lengths.""")
    return super().get_model_feature_lengths(task_feature_lengths)


@gin.configurable(module="feature_converters")
class NegativeAndPositiveTrainingFullFeatureConverter(
    NegativeAndPositiveTrainingFeatureConverterBase):
  """FeatureConverter variant for negative and positive examples.

  Weights given to negative and positive targets are full.

  Example:
  ds = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 1],
        "corrected_targets": [3]
    }, {
        "inputs": [6, 5, 4, 3],
        "negative_targets": [0],
        "corrected_targets": [1, 2, 3, 4]
    }]

  task_feature_lengths = {"inputs": 5, "targets": 4}

  converted_ds = [{
                     "inputs": [9, 4, 3, 8, 1],
      "decoder_target_tokens": [3, 9, 4, 1],
       "decoder_input_tokens": [0, 3, 9, 4],
       "decoder_loss_weights": [-1, -1, -1, -1],
        "decoder_segment_ids": [1, 1, 1, 1],
          "decoder_positions": [0, 1, 2, 0],
  },
  {
                     "inputs": [6, 5, 4, 3, 0],
      "decoder_target_tokens": [1, 2, 3, 4],
       "decoder_input_tokens": [0, 1, 2, 3],
       "decoder_loss_weights": [1, 1, 1, 1],
        "decoder_segment_ids": [1, 1, 1, 1],
          "decoder_positions": [0, 1, 2, 0],
  }]

  """

  NEGATIVE_FEATURE_CONVERTER_CLS = NegativeTrainingFullFeatureConverter
  POSITIVE_FEATURE_CONVERTER_CLS = CorrectiveTrainingFullFeatureConverter


@gin.configurable(module="feature_converters")
class NegativeAndPositiveTrainingFirstFeatureConverter(
    NegativeAndPositiveTrainingFeatureConverterBase):
  """FeatureConverter variant for negative and positive examples.

  Weights given to negative_targets is based on the first token diff with
  corrected_examples, while the weights given to positive targets are full.

  Example:
  ds = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 1],
        "corrected_targets": [3]
    }, {
        "inputs": [6, 5, 4, 3],
        "negative_targets": [0],
        "corrected_targets": [1, 2, 3, 4]
    }]

  task_feature_lengths = {"inputs": 5, "targets": 4}

  converted_ds = [{
                     "inputs": [9, 4, 3, 8, 1],
      "decoder_target_tokens": [3, 9, 4, 1],
       "decoder_input_tokens": [0, 3, 9, 4],
       "decoder_loss_weights": [0, -1, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 1],
          "decoder_positions": [0, 1, 2, 0],
  },
  {
                     "inputs": [6, 5, 4, 3, 0],
      "decoder_target_tokens": [1, 2, 3, 4],
       "decoder_input_tokens": [0, 1, 2, 3],
       "decoder_loss_weights": [1, 1, 1, 1],
        "decoder_segment_ids": [1, 1, 1, 1],
          "decoder_positions": [0, 1, 2, 0],
  }]

  """

  NEGATIVE_FEATURE_CONVERTER_CLS = NegativeTrainingFirstFeatureConverter
  POSITIVE_FEATURE_CONVERTER_CLS = CorrectiveTrainingFullFeatureConverter


@gin.configurable(module="feature_converters")
class NegativeAndPositiveTrainingDiffFeatureConverter(
    NegativeAndPositiveTrainingFeatureConverterBase):
  """FeatureConverter variant for negative and positive examples.

  Weights given to negative_targets is based on the diff with
  corrected_examples, while the weights given to positive targets are full.

  Example:
  ds = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 1],
        "corrected_targets": [3]
    }, {
        "inputs": [6, 5, 4, 3],
        "negative_targets": [0],
        "corrected_targets": [1, 2, 3, 4]
    }]

  task_feature_lengths = {"inputs": 5, "targets": 4}

  converted_ds = [{
                     "inputs": [9, 4, 3, 8, 1],
      "decoder_target_tokens": [3, 9, 4, 1],
       "decoder_input_tokens": [0, 3, 9, 4],
       "decoder_loss_weights": [0, -1, -1, -1],
        "decoder_segment_ids": [1, 1, 1, 1],
          "decoder_positions": [0, 1, 2, 0],
  },
  {
                     "inputs": [6, 5, 4, 3, 0],
      "decoder_target_tokens": [1, 2, 3, 4],
       "decoder_input_tokens": [0, 1, 2, 3],
       "decoder_loss_weights": [1, 1, 1, 1],
        "decoder_segment_ids": [1, 1, 1, 1],
          "decoder_positions": [0, 1, 2, 0],
  }]

  """

  NEGATIVE_FEATURE_CONVERTER_CLS = NegativeTrainingDiffFeatureConverter
  POSITIVE_FEATURE_CONVERTER_CLS = CorrectiveTrainingFullFeatureConverter
