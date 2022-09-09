"""Tests for models.

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
import dataclasses
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import seqio
import t5.data.tasks  # pylint:disable=unused-import
from t5patches import models
from t5patches import network
from t5patches import trainer as corrections_trainer_lib
from t5x import adafactor
from t5x import partitioning
from t5x import trainer as trainer_lib
from t5x import utils
import tensorflow as tf

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

PartitionSpec = partitioning.PartitionSpec
BATCH_SIZE, ENCODER_LEN, MAX_DECODE_LEN, EMBED_DIM = 2, 3, 4, 5


def get_t5_test_model(model_cls, **config_overrides):
  """Returns a tiny T5 1.1 model to use for testing.

  Near-copy of t5x.test_utils.get_t5_test_model.py, except that the function
  accepts a model_cls argument to determine the type of EncoderDecoder model
  used.

  Args:
    model_cls: a subclass of BaseModel, e.g. models.EncoderDecoderModelUL
    **config_overrides: any keyword arguments to override the T5Config
  Returns: an instance of model_cls with the config specified by the default
    config or config overrides.
  """
  tiny_config = network.T5Config(
      vocab_size=32128,
      dtype='bfloat16',
      emb_dim=8,
      num_heads=4,
      num_encoder_layers=2,
      num_decoder_layers=2,
      head_dim=3,
      mlp_dim=16,
      mlp_activations=('gelu', 'linear'),
      dropout_rate=0.0,
      logits_via_embedding=False,
  )

  tiny_config = dataclasses.replace(tiny_config, **config_overrides)
  sentencepiece_model_file = '/bigstore/t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
  vocabulary = seqio.SentencePieceVocabulary(sentencepiece_model_file)
  return model_cls(
      module=network.Transformer(tiny_config),
      input_vocabulary=vocabulary,
      output_vocabulary=vocabulary,
      optimizer_def=adafactor.Adafactor(
          decay_rate=0.8,
          step_offset=0,
          logical_factor_rules=adafactor.standard_logical_factor_rules()))


class EncoderDecoderModelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='unlikelihood_model',
          model_cls=models.EncoderDecoderModelUL,
          expected_scores=[-2.09588997, -1.3750219]),
      dict(
          testcase_name='negative_likelihood_model',
          model_cls=models.EncoderDecoderModelNL,
          expected_scores=[-0.40760607, 1.8152121]),
      dict(
          testcase_name='targeted_negative_model',
          model_cls=models.EncoderDecoderModelTN,
          expected_scores=[-3.2936196, -3.9873507]),
  )
  def test_score_batch(self, model_cls, expected_scores):
    encoder_input_tokens = jnp.ones((2, 3))
    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_loss_weights = jnp.array([[1, 1, -1, 0], [0, -1, 0, -1]])
    logits = jnp.arange(0, 24).reshape((2, 4, 3))
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = logits
    mock_transformer.dtype = jnp.float32

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    def mock_init(self):
      self.module = mock_transformer

    with mock.patch.object(model_cls, '__init__', new=mock_init):
      model = model_cls()
      if isinstance(model, models.EncoderDecoderModelTN):
        res = model.score_batch(params, batch, orig_params=params)
      else:
        res = model.score_batch(params, batch)

    mock_transformer.apply.assert_called_with({'params': params},
                                              encoder_input_tokens,
                                              decoder_input_tokens,
                                              decoder_target_tokens,
                                              encoder_segment_ids=None,
                                              decoder_segment_ids=None,
                                              encoder_positions=None,
                                              decoder_positions=None,
                                              decode=False,
                                              enable_dropout=False,
                                              rngs=None,
                                              mutable=False)
    # Scores are not log likelihood. Instead, they are log likelihood for
    # positive tokens and negative unlikelihood for negative tokens.
    np.testing.assert_allclose(res, expected_scores, rtol=1e-4)

  @parameterized.named_parameters(
      dict(
          testcase_name='unlikelihood_model',
          model_cls=models.EncoderDecoderModelUL,
          expected_scores=[-2.09588997, -1.3750219]),
      dict(
          testcase_name='negative_likelihood_model',
          model_cls=models.EncoderDecoderModelNL,
          expected_scores=[-0.40760607, 1.8152121]),
      dict(
          testcase_name='targeted_negative_model',
          model_cls=models.EncoderDecoderModelTN,
          expected_scores=[-3.2936196, -3.9873507]),
  )
  def test_score_batch_can_return_intermediates(self, model_cls,
                                                expected_scores):
    encoder_input_tokens = jnp.ones((2, 3))
    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_loss_weights = jnp.array([[1, 1, -1, 0], [0, -1, 0, -1]])
    logits = jnp.arange(0, 24).reshape((2, 4, 3))
    modified_variables = {'intermediates': {'bar': jnp.ones(5)}}
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = (logits, modified_variables)
    mock_transformer.dtype = jnp.float32

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    def mock_init(self):
      self.module = mock_transformer

    with mock.patch.object(model_cls, '__init__', new=mock_init):
      model = model_cls()
      if isinstance(model, models.EncoderDecoderModelTN):
        scores, intermediates = model.score_batch(
            params, batch, return_intermediates=True, orig_params=params)
      else:
        scores, intermediates = model.score_batch(
            params, batch, return_intermediates=True)

    mock_transformer.apply.assert_called_with({'params': params},
                                              encoder_input_tokens,
                                              decoder_input_tokens,
                                              decoder_target_tokens,
                                              encoder_segment_ids=None,
                                              decoder_segment_ids=None,
                                              encoder_positions=None,
                                              decoder_positions=None,
                                              decode=False,
                                              enable_dropout=False,
                                              rngs=None,
                                              mutable=['intermediates'])
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-4)
    # Incumbent intermediates are passed out unchanged.
    np.testing.assert_allclose(intermediates['bar'], jnp.ones(5))
    # A new collection of decoder intermediates are inserted by score_batch()
    np.testing.assert_allclose(intermediates['decoder']['loss_weights'][0],
                               decoder_loss_weights)
    np.testing.assert_allclose(intermediates['decoder']['target_tokens'][0],
                               decoder_target_tokens)

  @parameterized.named_parameters(
      dict(
          testcase_name='unlikelihood_model',
          model_cls=models.EncoderDecoderModelUL,
          trainer_cls=trainer_lib.Trainer),
      dict(
          testcase_name='negative_likelihood_model',
          model_cls=models.EncoderDecoderModelNL,
          trainer_cls=trainer_lib.Trainer),
      dict(
          testcase_name='targeted_negative_model',
          model_cls=models.EncoderDecoderModelTN,
          trainer_cls=corrections_trainer_lib.SelfDistillationTrainer),
  )
  def test_train_transformer_wmt(self, model_cls, trainer_cls):
    # Note: since this test initializes and trains three separate models for
    # a step, will run into a timeout error. To avoid this, pass flag
    # --test_timeout=600.
    # Dummy input data
    input_shape = (16, 8)
    encoder_input_tokens = np.ones(shape=input_shape, dtype=np.float32)
    decoder_input_tokens = 5 * np.ones(shape=input_shape, dtype=np.float32)
    decoder_target_tokens = 5 * np.ones(input_shape, dtype=np.float32)
    decoder_loss_weights = 5 * np.random.choice([-1, 1, 0], size=input_shape)
    # input_data = {'inputs': inputs, 'targets': targets}
    input_data = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights,
    }

    partitioner = partitioning.PjitPartitioner(
        num_partitions=1, use_cpu_pjit=True)

    model = get_t5_test_model(model_cls)

    ds_iter = tf.data.Dataset.from_tensors(input_data).as_numpy_iterator()
    input_shapes = {k: input_shape for k in input_data}

    train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        partitioner=partitioner)
    train_state_axes = train_state_initializer.train_state_axes
    train_state = train_state_initializer.from_scratch(jax.random.PRNGKey(0))

    trainer = trainer_cls(
        model,
        train_state=train_state,
        partitioner=partitioner,
        eval_names=[],
        summary_dir=None,
        train_state_axes=train_state_axes,
        rng=jax.random.PRNGKey(0),
        learning_rate_fn=lambda x: 0.001,
        num_microbatches=1)

    trainer.train(ds_iter, 1)
    logging.info('optimizer after first step %s', train_state.params)


if __name__ == '__main__':
  absltest.main()
