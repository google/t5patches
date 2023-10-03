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

"""T5X Models with alternative losses and scores to address negative tokens.

The primary purpose for these models is for their alternative loss functions
to use for model training. These models handle outputs with positive, negative,
and zero token weights.

The output of score_batch for these models are the negative per-sequence losses.
This choice was made to enable the additional functionality of obtaining
per-sequence contributions to the negative loss.

If one wishes to get per-sequence log likelihood scores, the model class should
be set to EncoderDecoderModel during inference or evaluation, which is still
possible even if one of these model classes was used during finetuning.

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
from typing import Any, Mapping, Optional, Tuple, Union

from flax import core as flax_core
from flax.training import common_utils
import gin
import jax
import jax.numpy as jnp
import numpy as np
from t5patches.feature_converters import NegativeTrainingFirstFeatureConverter
from t5x import losses
from t5x import metrics as metrics_lib
from t5x.models import EncoderDecoderModel
import tensorflow as tf

Array = Union[np.ndarray, jax.Array, tf.Tensor]
MetricsMap = metrics_lib.MetricsMap
PyTree = Any


@gin.configurable(module='models')
class EncoderDecoderModelNL(EncoderDecoderModel):
  """Negative training with negative likelihood.

  The loss for the tokens is simply multiplied by the weight. This means that
  we are trying to maximize the negative likelihood of tokens when they have
  negative weights. Otherwise, if a token is given a positive weight, typical
  training proceeds.

  The logic for this class is the same as EncoderDecoderModel except that in
  this class, we can optionally set a threshold after which we do not
  incentivize the log probabilities assigned to negative tokens to be pushed
  down further. This thresholding prevents the negative targets from continuing
  to affect the loss after they have already been sufficiently pushed down in
  probability to be very unlikely.

  This is a baseline for comparison with targeted negative training
  (EncoderDecoderModelTN).

  """

  FEATURE_CONVERTER_CLS = NegativeTrainingFirstFeatureConverter
  THRESHOLD = False
  # estimated to be 1000x less likely than the probability of a single token
  # in a 250k vocabulary if the distribution was uniform, i.e.
  # log(1/250k * .001)
  EPSILON = -19.3

  def _per_token_ce_nl(self, logits: jnp.ndarray, target_tokens: jnp.ndarray,
                       weights: jnp.ndarray) -> jnp.ndarray:
    """Return cross entropy loss per token, multiplied by weights.

    For positive weights, the per-token loss will simply be cross entropy,
    which encourages maximizing the likelihood of such tokens. For negative
    weights, the per-token loss will be negative likelihood, which encourages
    maximizing the negative likelihood of such tokens.

    Args:
      logits: [batch, seq, vocab] array of model logits
      target_tokens: [batch, seq] array of target tokens
      weights: [batch, seq] array of weights associated with each target token

    Returns:
      [batch, seq] array of per-token cross entropy scores multiplied by weights

    """
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0)

    ce_token_scores = losses.cross_entropy_with_logits(
        logits, targets, z_loss=0.0)[0] * weights

    # optionally threshold so there is no incentive to push down probabilities
    # further after a certain point
    if self.THRESHOLD:
      ce_token_scores = jnp.where(-ce_token_scores < self.EPSILON,
                                  -self.EPSILON, ce_token_scores)

    return ce_token_scores

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Loss function used for training with a negative likelihood.

    Tokens that are given a negative weight contribute to the loss in the
    opposite direction that they would for traditional maximum likelihood
    training.

    Does not use compute_weighted_cross_entropy so no z_loss or
    log_normalizing_factor.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.

    """
    logits = self._compute_logits(
        params, batch, dropout_rng=dropout_rng, mutable=False)
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    ce_token_scores = self._per_token_ce_nl(logits, target_tokens, weights)
    loss = jnp.sum(ce_token_scores)

    metrics = self._compute_metrics(
        logits=logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=None)

    return loss, metrics


@gin.configurable(module='models')
class EncoderDecoderModelUL(EncoderDecoderModel):
  """Negative training with unlikelihood.

  Tokens with a positive weight contribute to the loss via traditional cross
  entropy, or maximizing likelihood p(x_t | x_{<t}). Tokens with a
  negative weight contribute to the loss via unlikelihood, or maximizing
  log(1 - p(x_t | x_{<t})).

  This is a baseline for comparison with targeted negative training
  (EncoderDecoderModelTN).

  """

  FEATURE_CONVERTER_CLS = NegativeTrainingFirstFeatureConverter

  def _per_token_ce_ul(self, logits: jnp.ndarray, target_tokens: jnp.ndarray,
                       weights: jnp.ndarray) -> jnp.ndarray:
    """Return a per-token cross entropy or negative unlikelihood loss.

    Positive tokens contribute a cross entropy term to the loss. Negative tokens
    contribute a negative unlikelihood loss, i.e. log(1 - p(x_t|x_{<t})). Thus,
    the loss encourages maximizing the likelihood of positive tokens (or
    minimizing their cross entropy) and maximizing the unlikelihood of negative
    tokens (or minimizing their negative unlikelihood).

    Args:
      logits: logits
      target_tokens: target_tokens
      weights: weights

    Returns:
      per_token_loss: the loss computed for the each token [batch, len]


    """
    positive_weights_mask = jnp.where(weights > 0, 1, 0)
    negative_weights_mask = jnp.where(weights < 0, 1, 0)
    # Purposefully don't use config.z_loss because that term is for training
    # stability and shouldn't affect our reported scores.
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0)

    log_probas = -losses.cross_entropy_with_logits(
        logits, targets, z_loss=0.0)[0]
    ce_token_scores = jnp.where(weights > 0,
                                -log_probas * positive_weights_mask, 0)

    eps = 1e-7
    # ul = log (1 - p)
    # we wish to minimize negative ul in the same way we wish to minimize ce
    ul_token_scores = jnp.where(
        weights < 0, -jnp.log(1.0 - jnp.clip(jnp.exp(log_probas), 0, 1 - eps)) *
        negative_weights_mask, 0)

    return ce_token_scores + ul_token_scores

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute score (negative loss) on a batch.

    Differs from `score_batch` of its parent, EncoderDecoderModel, in that the
    parent class scores are negative cross entropy, i.e. log likelihood, while
    here, the score is the negative of the loss, i.e. negative cross entropy
    (or log likelihood) for positive tokens, unlikelihood for negative tokens,
    and zero otherwise.

    Args:
      params: model parameters
      batch: data batch of size [batch, seq, vocab]
      return_intermediates: whether to return intermediate values

    Returns:
      Either an array of scores of size [batch], if return_intermediates is
      False, or a tuple of scores and intermediate values, if
      return_intermediates is True.

    """
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    if return_intermediates:
      logits, modified_variables = self._compute_logits(
          params=params, batch=batch, mutable=['intermediates'])

      # Inside self.module, we called nn.Module.sow to track various
      # intermediate values. We extract them here.
      intermediates = flax_core.unfreeze(
          modified_variables.get('intermediates', {}))

      # Track per-token labels and loss weights as well. These are not
      # intermediate values of logit computation, so we manually add them here.
      intermediates.setdefault('decoder', {})
      intermediates['decoder']['target_tokens'] = (target_tokens,)
      intermediates['decoder']['loss_weights'] = (weights,)
      # Note that the values are singleton tuples. This is because values inside
      # `intermediates` should be tuples tracking all instantiations of a value.
      # These values each have just one instantiation, hence singletons.
    else:
      logits = self._compute_logits(params, batch)  # type: jnp.ndarray  # pytype: disable=annotation-type-mismatch  # jax-ndarray

    per_token_ce_ul = self._per_token_ce_ul(logits, target_tokens, weights)
    sequence_scores = per_token_ce_ul.sum(-1) * -1  # [batch]

    if return_intermediates:
      return sequence_scores, intermediates

    return sequence_scores

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Unlikelihood-based loss function used for training.

    The loss is the summation of cross entropy loss for positive tokens and
    negative unlikelihood for negative tokens. See `_per_token_ce_ul` docstring
    for more information.

    Does not use compute_weighted_cross_entropy so no support for z_loss.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      metrics: a mapping of metrics computed for this batch.

    """
    logits = self._compute_logits(params, batch, dropout_rng, mutable=False)

    (_, weights) = losses.get_loss_normalizing_factor_and_weights(
        self._loss_normalizing_factor, batch)

    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    per_token_ce_ul = self._per_token_ce_ul(logits, target_tokens,
                                            weights)  # [batch, sequence]

    loss = jnp.sum(per_token_ce_ul)

    metrics = self._compute_metrics(
        logits=logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=None)
    return loss, metrics


class SelfDistillationEncoderDecoderModel(EncoderDecoderModel, abc.ABC):
  """Encoder-decoder that incorporates original model predictions in the loss.

  Should be used in conjunction with the SelfDistillationTrainer in trainer.py.

  """

  @abc.abstractmethod
  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
      orig_params: PyTree,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Computes loss during training.

    Same logic as `loss_fn` in the parent EncoderDecoderModel class, except that
    orig_params are also accepted as input.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.
      orig_params: model parameters of the original model.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """

  def eval_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      orig_params: PyTree,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Computes loss and metrics during the evaluation.

    Same logic as `eval_fn` in the parent EncoderDecoderModel class, except that
    orig_params are also accepted as input.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      orig_params: model parameters of the original model.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    return self.loss_fn(
        params=params,
        batch=batch,
        dropout_rng=None,
        orig_params=orig_params,
    )


@gin.configurable(module='models')
class EncoderDecoderModelTN(SelfDistillationEncoderDecoderModel):
  """Negative training with targeted correction (TC).

  The loss is cross entropy between the current model distribution and the
  desired one where the negative token has near-zero probability. The resulting
  probability distribution is renormalized.

  """

  FEATURE_CONVERTER_CLS = NegativeTrainingFirstFeatureConverter

  def _get_desired_dist(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Get the desired conditional distributions to optimize towards.

    The desired distribution is either the original probability distribution,
    if no weight is associated with the output token index, or a probability
    distribution where the particular output token is near-zero probability,
    if a negative weight is associated with the output token index.

    Args:
      logits: model logits [batch, seq, vocab]
      target_tokens: target tokens, token idx only [batch, seq]
      weights: weights associated with the target tokens [batch, seq]
      orig_logits: logits from the original model before any training [batch,
        seq, vocab]

    Returns:
      A [batch, seq, vocab] array representing the desired distribution.
    """
    # target_tokens is [batch, seq]. targets is [batch, seq, num_classes]
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0)

    original_dist = jax.nn.softmax(orig_logits, axis=-1)
    # zero out the probability at a target token if it is assigned a -1 weight
    # return original_dist otherwise
    desired_dist_unnorm = jnp.where(
        jnp.expand_dims(weights, -1) == -1,
        original_dist - targets * original_dist,
        original_dist,
    )

    desired_dist = (
        desired_dist_unnorm / desired_dist_unnorm.sum(-1, keepdims=True))
    return desired_dist

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
      orig_params: PyTree,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Minimize cross entropy between the model and desired distributions.

    Targeted negative training considers each conditional distribution
    p_m(x_t | x_{<t}) in the model output and optimizes each of these
    distributions to approximate a desired distribution, calculated based on
    the original model output probabilities at the beginning of training and
    the negative targets provided.

    For a given negative token, the desired distribution p_d(x_t | x_{<t}) is
    proportional to the original distribution p_o(x_t | x_{<t}) with the
    particular target token's probability set to zero. For tokens with weight 0,
    the desired distribution is set to the original model distribution:
    p_d(x_t | x_{<t}) = p_o(x_t | x_{<t}).

    The loss function minimizes the cross entropy between the desired
    distribution and the current model distribution for each of the token
    output distributions. This is analogous to minimizing the KL divergence
    between the desired distribution and the model distribution.

    Args:
      params: model params.
      batch: data batch.
      dropout_rng: rng to use for dropout, or None for deterministic mode.
      orig_params: original model params.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.

    """
    logits = self._compute_logits(
        params, batch, dropout_rng=dropout_rng, mutable=False)

    # logits = self._compute_logits(params, batch)
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    orig_logits = self._compute_logits(
        orig_params, batch, dropout_rng=dropout_rng, mutable=False)

    desired_dist = self._get_desired_dist(logits, target_tokens, weights,
                                          orig_logits)

    mask = jnp.where(weights != 0, 1, 0)
    ce_token_scores = (
        losses.cross_entropy_with_logits(logits, desired_dist, z_loss=0.0)[0]
        * mask
    )

    loss = jnp.sum(ce_token_scores)

    metrics = self._compute_metrics(
        logits=logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=None)

    return loss, metrics

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      orig_params: PyTree = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute score (negative loss) on a batch.

    Differs from `score_batch` of its parent, EncoderDecoderModel, in that the
    former is negative cross entropy between the data and model distribution,
    i.e. log likelihood, whereas this score is the negative cross entropy
    between the model distributions and the desired distributions, either the
    original distribution, if the token weight is zero, or proportional to the
    original distribution with the target token probability zeroed out, if the
    token weight is negative.

    Args:
      params: model parameters
      batch: data batch of size [batch, seq, vocab]
      return_intermediates: whether to return intermediate values
      orig_params: original model parameters at the start of training

    Returns:
      Either an array of scores of size [batch], if return_intermediates is
      False, or a tuple of scores and intermediate values, if
      return_intermediates is True.

    """
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    if return_intermediates:
      logits, modified_variables = self._compute_logits(
          params=params, batch=batch, mutable=['intermediates'])
      orig_logits, _ = self._compute_logits(
          orig_params, batch, mutable=['intermediates'])

      # Inside self.module, we called nn.Module.sow to track various
      # intermediate values. We extract them here.
      intermediates = flax_core.unfreeze(
          modified_variables.get('intermediates', {}))

      # Track per-token labels and loss weights as well. These are not
      # intermediate values of logit computation, so we manually add them here.
      intermediates.setdefault('decoder', {})
      intermediates['decoder']['target_tokens'] = (target_tokens,)
      intermediates['decoder']['loss_weights'] = (weights,)
      # Note that the values are singleton tuples. This is because values inside
      # `intermediates` should be tuples tracking all instantiations of a value.
      # These values each have just one instantiation, hence singletons.
    else:
      logits = self._compute_logits(params, batch)  # type: jnp.ndarray  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      orig_logits = self._compute_logits(orig_params, batch)
    desired_dist = self._get_desired_dist(logits, target_tokens, weights,
                                          orig_logits)

    mask = jnp.where(weights != 0, 1, 0)
    ce_token_scores = (
        losses.cross_entropy_with_logits(logits, desired_dist, z_loss=0.0)[0]
        * mask
    )

    sequence_scores = jnp.sum(ce_token_scores, axis=-1) * -1  # [batch]

    if return_intermediates:
      return sequence_scores, intermediates

    return sequence_scores
