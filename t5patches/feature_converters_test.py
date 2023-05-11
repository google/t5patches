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

"""Tests for fcs.

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

from absl.testing import absltest
from absl.testing import parameterized
import seqio
from t5patches import feature_converters as fcs
import tensorflow.compat.v2 as tf


tf.compat.v1.enable_eager_execution()

assert_dataset = seqio.test_utils.assert_dataset
create_default_dataset = seqio.test_utils.create_default_dataset


class NegativeTrainingFeatureConverterTest(parameterized.TestCase):
  params_ntd = dict(
      testcase_name="ntd",
      feature_converter=fcs.NegativeTrainingDiffFeatureConverter(pack=False),
      expected_weights=[1, 1, -1, -1, 0],
      decoder_tokens=[3, 9, 4, 1, 0],
  )
  params_nt1 = dict(
      testcase_name="nt1",
      feature_converter=fcs.NegativeTrainingFirstFeatureConverter(pack=False),
      expected_weights=[1, 1, -1, 1, 0],
      decoder_tokens=[3, 9, 4, 1, 0],
  )
  params_ntf = dict(
      testcase_name="ntf",
      feature_converter=fcs.NegativeTrainingFullFeatureConverter(pack=False),
      expected_weights=[-1, -1, -1, -1, 0],
      decoder_tokens=[3, 9, 4, 1, 0],
  )
  params_ctd = dict(
      testcase_name="ctd",
      feature_converter=fcs.CorrectiveTrainingDiffFeatureConverter(pack=False),
      expected_weights=[0, 0, 1, 1, 0],
      decoder_tokens=[3, 9, 3, 2, 0],
  )
  params_ct1 = dict(
      testcase_name="ct1",
      feature_converter=fcs.CorrectiveTrainingFirstFeatureConverter(pack=False),
      expected_weights=[0, 0, 1, 0, 0],
      decoder_tokens=[3, 9, 3, 2, 0],
  )
  params_ctf = dict(
      testcase_name="ctf",
      feature_converter=fcs.CorrectiveTrainingFullFeatureConverter(pack=False),
      expected_weights=[1, 1, 1, 1, 0],
      decoder_tokens=[3, 9, 3, 2, 0],
  )

  @parameterized.named_parameters(
      params_ntd,
      params_nt1,
      params_ntf,
      params_ctd,
      params_ct1,
      params_ctf,
  )
  def test_negative_training_unpacked(self, feature_converter, expected_weights,
                                      decoder_tokens):
    x = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 1],
        "corrected_targets": [3, 9, 3, 2]
    }]
    ds = create_default_dataset(
        x, feature_names=("inputs", "negative_targets", "corrected_targets"))
    task_feature_lengths = {
        "inputs": 7,
        "negative_targets": 5,
        "corrected_targets": 5
    }
    converted_ds = feature_converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_tokens": decoder_tokens,
        "decoder_input_tokens": [0] + decoder_tokens[:-1],
        "decoder_loss_weights": expected_weights,
    }
    assert_dataset(converted_ds, expected)

  params_ntd_max = dict(
      testcase_name="ntd",
      feature_converter=fcs.NegativeTrainingDiffFeatureConverter(pack=False),
      expected_weights=[1, 1, -1, -1, -1],
      decoder_tokens=[3, 9, 4, 5, 1],
  )
  params_nt1_max = dict(
      testcase_name="nt1",
      feature_converter=fcs.NegativeTrainingFirstFeatureConverter(pack=False),
      expected_weights=[1, 1, -1, 1, 1],
      decoder_tokens=[3, 9, 4, 5, 1],
  )
  params_ntf_max = dict(
      testcase_name="ntf",
      feature_converter=fcs.NegativeTrainingFullFeatureConverter(pack=False),
      expected_weights=[-1, -1, -1, -1, -1],
      decoder_tokens=[3, 9, 4, 5, 1],
  )
  params_ctd_max = dict(
      testcase_name="ctd",
      feature_converter=fcs.CorrectiveTrainingDiffFeatureConverter(pack=False),
      expected_weights=[0, 0, 1, 1, 0],
      decoder_tokens=[3, 9, 3, 1, 0],
  )
  params_ct1_max = dict(
      testcase_name="ct1",
      feature_converter=fcs.CorrectiveTrainingFirstFeatureConverter(pack=False),
      expected_weights=[0, 0, 1, 0, 0],
      decoder_tokens=[3, 9, 3, 1, 0],
  )
  params_ctf_max = dict(
      testcase_name="ctf",
      feature_converter=fcs.CorrectiveTrainingFullFeatureConverter(pack=False),
      expected_weights=[1, 1, 1, 1, 0],
      decoder_tokens=[3, 9, 3, 1, 0],
  )

  @parameterized.named_parameters(
      params_ntd_max,
      params_nt1_max,
      params_ntf_max,
      params_ctd_max,
      params_ct1_max,
      params_ctf_max,
  )
  def test_negative_training_targets_max_length(self, feature_converter,
                                                expected_weights,
                                                decoder_tokens):
    x = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 5, 1],
        "corrected_targets": [3, 9, 3, 1]
    }]
    ds = create_default_dataset(
        x, feature_names=("inputs", "negative_targets", "corrected_targets"))
    task_feature_lengths = {
        "inputs": 5,
        "negative_targets": 5,
        "corrected_targets": 5
    }
    converted_ds = feature_converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1],
        "decoder_target_tokens": decoder_tokens,
        "decoder_input_tokens": [0] + decoder_tokens[:-1],
        "decoder_loss_weights": expected_weights,
    }
    assert_dataset(converted_ds, expected)

  params_nt1equal = dict(
      testcase_name="nt1equal",
      feature_converter=fcs.NegativeTrainingFirstFeatureConverter(pack=False),
      expected_weights=[1, 1, 1, 1, 1],
      decoder_tokens=[3, 9, 4, 5, 1],
  )
  params_ct1equal = dict(
      testcase_name="ct1equal",
      feature_converter=fcs.CorrectiveTrainingFirstFeatureConverter(pack=False),
      expected_weights=[0, 0, 0, 0, 0],
      decoder_tokens=[3, 9, 4, 5, 1],
  )

  @parameterized.named_parameters(
      params_nt1equal,
      params_ct1equal,
  )
  def test_negative_training_targets_equal(self, feature_converter,
                                           expected_weights, decoder_tokens):
    x = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 5, 1],
        "corrected_targets": [3, 9, 4, 5, 1]
    }]
    ds = create_default_dataset(
        x, feature_names=("inputs", "negative_targets", "corrected_targets"))
    task_feature_lengths = {
        "inputs": 5,
        "negative_targets": 5,
        "corrected_targets": 5
    }
    converted_ds = feature_converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1],
        "decoder_target_tokens": decoder_tokens,
        "decoder_input_tokens": [0] + decoder_tokens[:-1],
        "decoder_loss_weights": expected_weights,
    }
    assert_dataset(converted_ds, expected)

  params_ntd_long = dict(
      testcase_name="ntd",
      feature_converter=fcs.NegativeTrainingDiffFeatureConverter(pack=False),
  )
  params_nt1_long = dict(
      testcase_name="nt1",
      feature_converter=fcs.NegativeTrainingFirstFeatureConverter(pack=False),
  )
  params_ntf_long = dict(
      testcase_name="ntf",
      feature_converter=fcs.NegativeTrainingFullFeatureConverter(pack=False),
  )
  params_ctd_long = dict(
      testcase_name="ctd",
      feature_converter=fcs.CorrectiveTrainingDiffFeatureConverter(pack=False),
  )
  params_ct1_long = dict(
      testcase_name="ct1",
      feature_converter=fcs.CorrectiveTrainingFirstFeatureConverter(pack=False),
  )
  params_ctf_long = dict(
      testcase_name="ctf",
      feature_converter=fcs.CorrectiveTrainingFullFeatureConverter(pack=False),
  )

  @parameterized.named_parameters(
      params_ntd_long,
      params_nt1_long,
      params_ntf_long,
      params_ctd_long,
      params_ct1_long,
      params_ctf_long,
  )
  def test_negative_training_extra_long_inputs(self, feature_converter):
    x = [{
        "inputs": [9, 4, 3, 8, 4, 5, 1],
        "negative_targets": [3, 9, 4, 7, 8, 1],
        "corrected_targets": []
    }]
    ds = create_default_dataset(
        x, feature_names=("inputs", "negative_targets", "corrected_targets"))
    task_feature_lengths = {
        "inputs": 5,
        "negative_targets": 8,
        "corrected_targets": 0
    }
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 5 during input_validation.*")

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, expected_msg):
      converted_ds = feature_converter(ds, task_feature_lengths)
      list(converted_ds.as_numpy_iterator())

  params_nptd = dict(
      testcase_name="nptd",
      feature_converter=fcs.NegativeAndPositiveTrainingDiffFeatureConverter(
          pack=False
      ),
      expected_weights=[[1, -1, -1, -1, 0], [1, 1, 1, 1, 0]],
  )
  params_npt1 = dict(
      testcase_name="npt1",
      feature_converter=fcs.NegativeAndPositiveTrainingFirstFeatureConverter(
          pack=False
      ),
      expected_weights=[[1, -1, 1, 1, 0], [1, 1, 1, 1, 0]],
  )
  params_nptf = dict(
      testcase_name="nptf",
      feature_converter=fcs.NegativeAndPositiveTrainingFullFeatureConverter(
          pack=False
      ),
      expected_weights=[[-1, -1, -1, -1, 0], [1, 1, 1, 1, 0]],
  )

  @parameterized.named_parameters(
      params_nptd,
      params_npt1,
      params_nptf,
  )
  def test_negative_and_positive_training_unpacked(self, feature_converter,
                                                   expected_weights):
    x = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 1],
        "corrected_targets": [3]
    }, {
        "inputs": [6, 5, 4, 3],
        "negative_targets": [0],
        "corrected_targets": [1, 2, 3, 4]
    }]
    ds = create_default_dataset(
        x, feature_names=("inputs", "negative_targets", "corrected_targets"))
    task_feature_lengths = {
        "inputs": 7,
        "negative_targets": 5,
        "corrected_targets": 5
    }
    converted_ds = feature_converter(ds, task_feature_lengths)

    expected = [{
        "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_tokens": [3, 9, 4, 1, 0],
        "decoder_input_tokens": [0, 3, 9, 4, 1],
        "decoder_loss_weights": expected_weights[0],
    }, {
        "encoder_input_tokens": [6, 5, 4, 3, 0, 0, 0],
        "decoder_target_tokens": [1, 2, 3, 4, 0],
        "decoder_input_tokens": [0, 1, 2, 3, 4],
        "decoder_loss_weights": expected_weights[1],
    }]
    assert_dataset(converted_ds, expected)

  params_nptd_unmatched = dict(
      testcase_name="nptd",
      feature_converter=fcs.NegativeAndPositiveTrainingDiffFeatureConverter(
          pack=False
      ),
  )
  params_npt1_unmatched = dict(
      testcase_name="npt1",
      feature_converter=fcs.NegativeAndPositiveTrainingFirstFeatureConverter(
          pack=False
      ),
  )
  params_nptf_unmatched = dict(
      testcase_name="nptf",
      feature_converter=fcs.NegativeAndPositiveTrainingFullFeatureConverter(
          pack=False
      ),
  )

  @parameterized.named_parameters(
      params_nptd_unmatched,
      params_npt1_unmatched,
      params_nptf_unmatched,
  )
  def test_negative_and_positive_unmatched_lengths(self, feature_converter):
    x = [{
        "inputs": [9, 4, 3, 8, 1],
        "negative_targets": [3, 9, 4, 1],
        "corrected_targets": [3]
    }, {
        "inputs": [6, 5, 4, 3],
        "negative_targets": [0],
        "corrected_targets": [1, 2, 3, 4]
    }]
    ds = create_default_dataset(
        x, feature_names=("inputs", "negative_targets", "corrected_targets"))
    task_feature_lengths = {
        "inputs": 7,
        "negative_targets": 5,
        "corrected_targets": 4
    }

    expected_msg = (
        r".*Negative and corrected targets should be assigned the same lengths.*"
    )

    with self.assertRaisesRegex(ValueError, expected_msg):
      converted_ds = feature_converter(ds, task_feature_lengths)
      list(converted_ds.as_numpy_iterator())


if __name__ == "__main__":
  absltest.main()
