# Copyright 2024 The T5Patches Authors.
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
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from flax import core as flax_core
from flax.core import scope as flax_scope
from flax.training import common_utils
import gin
import jax
import jax.numpy as jnp
import numpy as np
from t5x import losses
from t5x import metrics as metrics_lib
from t5x.models import BaseTransformerModel, EncoderDecoderModel
import tensorflow as tf

Array = Union[np.ndarray, jax.Array, tf.Tensor]
MetricsMap = metrics_lib.MetricsMap
PyTree = Any

ParamAxesNamesOverrides = Sequence[Tuple[str, Tuple[str, ...]]]
ParamAxesNamesOverrideFn = Callable[[], ParamAxesNamesOverrides]


class NegativeTrainingTransformer(BaseTransformerModel, abc.ABC):
  """Base class for models that use negative training losses."""

  def __init__(
      self,
      feature_converter_cls: Optional[Any] = None,
      alpha: float = 1e4,
      **kwargs: Any,
  ):
    if feature_converter_cls is not None:
      self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    self.alpha = alpha
    super().__init__(**kwargs)


class NLModel(NegativeTrainingTransformer, abc.ABC):
  """Transformer model with negative likelihood loss and score fn.

  The loss for the tokens is simply multiplied by the weight. This means that
  we are trying to maximize the negative likelihood of tokens when they have
  negative weights. Otherwise, if a token is given a positive weight, typical
  training proceeds.

  One can optionally set a threshold after which we do not
  incentivize the log probabilities assigned to negative tokens to be pushed
  down further. This thresholding prevents the negative targets from continuing
  to affect the loss after they have already been sufficiently pushed down in
  probability to be very unlikely.
  """

  THRESHOLD = True
  EPSILON = -10.3

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

    neg_mask = jnp.where(weights < 0, 1, 0)
    pos_mask = jnp.where(weights >= 0, 1, 0)

    ce_token_scores = (
        losses.cross_entropy_with_logits(logits, targets, z_loss=0.0)[0]
        * weights
        * (neg_mask * self.alpha + pos_mask)
    )

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
      dropout_rng: Optional[jax.Array],
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
    logits = self._compute_logits(  # pytype: disable=wrong-keyword-args
        params, batch, dropout_rng=dropout_rng, mutable=False
    )
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

  def score_batch(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute sequence-levels scores equal to the negative loss.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      return_intermediates: boolean for whether to return intermediate values.

    Returns:
      sequence_scores: sum of the log likelihoods (for negative tokens) and
      negative log likelihoods (for positive tokens) in a sequence. Note that
      score is the negation of the loss (higher is better).
      intermediates (optional): intermediate values from the forward pass.
    """
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    if return_intermediates:
      logits, modified_variables = self._compute_logits(  # pytype: disable=wrong-keyword-args
          params=params, batch=batch, mutable=['intermediates']
      )

      # Inside self.module, we called nn.Module.sow to track various
      # intermediate values. We extract them here.
      intermediates = flax_core.unfreeze(
          modified_variables.get('intermediates', {})
      )

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

    # Purposefully don't use config.z_loss because that term is for training
    # stability and shouldn't affect our reported scores.
    token_scores = self._per_token_ce_nl(logits, target_tokens, weights)

    if return_intermediates:
      intermediates['decoder']['token_scores'] = (token_scores,)

    sequence_scores = token_scores.sum(-1) * -1

    if return_intermediates:
      return sequence_scores, intermediates

    return sequence_scores


class ULModel(NegativeTrainingTransformer, abc.ABC):
  """Transformer model trained with unlikelihood.

  Tokens with a positive weight contribute to the loss via traditional cross
  entropy, or maximizing likelihood p(x_t | x_{<t}). Tokens with a
  negative weight contribute to the loss via unlikelihood, or maximizing
  log(1 - p(x_t | x_{<t})).
  """

  def _per_token_ce_ul(self, logits: jnp.ndarray, target_tokens: jnp.ndarray,
                       weights: jnp.ndarray) -> jnp.ndarray:
    """Return a per-token cross entropy or negative unlikelihood loss.

    Positive tokens contribute a cross entropy term to the loss. Negative tokens
    contribute a negative unlikelihood loss, where unlikelihood is
    log(1 - p(x_t|x_{<t})). Thus, the loss encourages maximizing the likelihood
    of positive tokens (or minimizing their cross entropy) and maximizing the
    unlikelihood of negative tokens (or minimizing their negative unlikelihood).

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
    ul_token_scores = (
        jnp.where(
            weights < 0,
            -jnp.log(1.0 - jnp.clip(jnp.exp(log_probas), 0, 1 - eps))
            * negative_weights_mask,
            0,
        )
        * self.alpha
    )

    return ce_token_scores + ul_token_scores

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute sequence-level scores (negative loss) on a batch.

    Differs from `score_batch` of its parent, BaseTransformerModel, in that
    scores in the parent class scores are log likelihood (i.e., negative cross
    entropy) multiplied by the weights, while here, the score is the negative
    of the loss.

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
      logits, modified_variables = self._compute_logits(  # pytype: disable=wrong-keyword-args
          params=params, batch=batch, mutable=['intermediates']
      )

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
      dropout_rng: Optional[jax.Array],
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
    logits = self._compute_logits(params, batch, dropout_rng, mutable=False)  # pytype: disable=wrong-keyword-args

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


@gin.configurable(module='models')
class SelfDistillationModel(NegativeTrainingTransformer, abc.ABC):
  """Transformer model that incorporates original model predictions in the loss.

  Models subclassed from this class should be used in conjunction with the
  SelfDistillationTrainer in trainer.py.
  """

  def __init__(
      self,
      alpha: float = 1e4,
      **kwargs: Any,
  ):
    self.alpha = alpha
    super().__init__(**kwargs)

  @abc.abstractmethod
  def _get_token_losses(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute token-level losses."""
    pass

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array],
      orig_params: PyTree = None,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Computes loss during training.

    Same logic as `loss_fn` in the ancestral BaseTransformerModel class, except
    that orig_params are also accepted as input.

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
    logits = self._compute_logits(  # pytype: disable=wrong-keyword-args
        params, batch, dropout_rng=dropout_rng, mutable=False
    )

    # logits = self._compute_logits(params, batch)
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']
    orig_logits = self._compute_logits(  # pytype: disable=wrong-keyword-args
        orig_params, batch, dropout_rng=dropout_rng, mutable=False
    )

    ce_token_scores = self._get_token_losses(
        logits, target_tokens, weights, orig_logits
    )

    loss = jnp.sum(ce_token_scores)

    metrics = self._compute_metrics(
        logits=logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=None,
    )

    return loss, metrics

  def eval_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      orig_params: PyTree,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Computes loss and metrics during the evaluation.

    Same logic as `eval_fn` in the ancestral BaseModel class, except that
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

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      orig_params: PyTree = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute score (negative loss) on a batch.

    The score is the negation of the loss. Higher is better.

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
      logits, modified_variables = self._compute_logits(  # pytype: disable=wrong-keyword-args
          params=params, batch=batch, mutable=['intermediates']
      )
      orig_logits, _ = self._compute_logits(  # pytype: disable=wrong-keyword-args
          orig_params, batch, mutable=['intermediates']
      )

      # Inside self.module, we called nn.Module.sow to track various
      # intermediate values. We extract them here.
      intermediates = flax_core.unfreeze(
          modified_variables.get('intermediates', {})
      )

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
    ce_token_scores = self._get_token_losses(
        logits, target_tokens, weights, orig_logits
    )
    sequence_scores = jnp.sum(ce_token_scores, axis=-1) * -1  # [batch]

    if return_intermediates:
      return sequence_scores, intermediates

    return sequence_scores


class TNFFModel(SelfDistillationModel, abc.ABC):
  """Targeted negative training with forward KL on positive and negative tokens.

  Targeted negative training considers each conditional distribution
  p_m(x_t | x_{<t}) in the model output and optimizes each of these
  distributions to approximate a desired distribution, calculated based on
  the original model output probabilities at the beginning of training and
  the negative targets provided.

  The target distribution for each token conditional is proportional to the
  original distribution with the negative tokens set to zero probability.

  The loss for TNFF is cross entropy between the current model distribution and
  this target distribution. Minimizing this cross-entropy is the same as
  minimizing the forward KL divergence between the target distribution and
  current model distribution, given the original model distribution entropy is
  fixed).
  """

  def _get_desired_dist(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Return the desired conditional distributions to optimize towards.

    The desired distribution is either the original probability distribution,
    if a positive weight is associated with the output token index, or a
    distribution where the output token is set to zero probability,
    if a negative weight is associated with the output token index.

    Args:
      logits: model logits [batch, seq, vocab]
      target_tokens: target tokens, token idx only [batch, seq]
      weights: weights associated with the target tokens [batch, seq]
      orig_logits: logits from the original model before any training [batch,
        seq, vocab]

    Returns:
      A [batch, seq, vocab] array of the desired distribution.
    """
    # target_tokens is [batch, seq]. targets is [batch, seq, num_classes]
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0)

    original_dist = jax.nn.softmax(orig_logits, axis=-1)
    # zero out the probability at a target token if it is assigned a -1 weight
    # return original_dist otherwise
    desired_dist_unnorm = jnp.where(
        jnp.expand_dims(weights, -1) > 0,
        original_dist,
        original_dist + targets * jnp.expand_dims(weights, -1) * original_dist,
    )

    desired_dist = (
        desired_dist_unnorm / desired_dist_unnorm.sum(-1, keepdims=True))
    return desired_dist

  def _get_token_losses(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute token-level losses."""
    desired_dist = self._get_desired_dist(
        logits, target_tokens, weights, orig_logits
    )
    neg_mask = jnp.where(weights < 0, 1, 0)
    pos_mask = jnp.where(weights > 0, 1, 0)
    ce_token_scores = losses.cross_entropy_with_logits(
        logits, desired_dist, z_loss=0.0
    )[0] * (neg_mask * self.alpha + pos_mask)
    return ce_token_scores


class TNRRModel(SelfDistillationModel, abc.ABC):
  """Targeted negative training with reverse KL divergence terms.

  The optimal distributions are the same as those for TNFFModel, i.e., original
  model distribution for positive indices and original distribution with
  negative tokens zeroed out for negative indices, but the loss is the reverse
  KL divergence rather than the forward KL divergence.
  """

  def _get_desired_logits(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Get the desired logits to optimize towards.

    The desired distribution is either the original probability distribution,
    if no weight is associated with the output token index, or a probability
    distribution where the particular output token is near-zero probability,
    if a negative weight is associated with the output token index.

    We return logits instead of the normalized distribution since the function
    losses.cross_entropy_with_logits() expects logits and performs normalization
    within the function itself.

    Args:
      logits: model logits [batch, seq, vocab]
      target_tokens: target tokens, token idx only [batch, seq]
      weights: weights associated with the target tokens [batch, seq]
      orig_logits: logits from the original model before any training [batch,
        seq, vocab]

    Returns:
      A [batch, seq, vocab] array of the desired logits.
    """
    # target_tokens is [batch, seq]. targets is [batch, seq, num_classes]
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0
    )

    # zero out the probability at a target token if it is assigned a -1 weight
    # return original_dist if the token is assigned a 1 weight
    # since we're working with logits, the logits become:
    # original logits if the weight is 1
    # -infty at the target token and original logits elsewhere if weight is -1
    large_number = 1e9
    desired_logits = jnp.where(
        jnp.expand_dims(weights, -1) > 0,
        orig_logits,
        orig_logits + targets * jnp.expand_dims(weights, -1) * large_number,
    )

    # convert logits to a smoothed version so the resulting desired dist does
    # not have zeros and avoid problems when we take logs
    desired_dist = jax.nn.softmax(desired_logits)
    desired_dist_smoothed_unnorm = desired_dist + 1e-6
    desired_dist_smoothed = desired_dist_smoothed_unnorm / jnp.sum(
        desired_dist_smoothed_unnorm, axis=-1, keepdims=True
    )
    desired_logits_smoothed = jnp.log(desired_dist_smoothed)
    return desired_logits_smoothed

  def _get_token_losses(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute token-level losses."""
    desired_logits = self._get_desired_logits(
        logits, target_tokens, weights, orig_logits
    )
    neg_mask = jnp.where(weights < 0, 1, 0)
    pos_mask = jnp.where(weights > 0, 1, 0)
    ce_token_scores = (
        losses.cross_entropy_with_logits(
            desired_logits, jax.nn.softmax(logits, axis=-1), z_loss=0.0
        )[0]
        - losses.cross_entropy_with_logits(
            logits, jax.nn.softmax(logits, axis=-1), z_loss=0.0
        )[0]
    ) * (neg_mask * self.alpha + pos_mask)
    return ce_token_scores  # [batch, seq]


class TNRFModel(SelfDistillationModel, abc.ABC):
  """Targeted negative training with reverse and forward KL divergence terms.

  The reverse KL is computed on indices with negative tokens, and the forward
  KL is computed on indices with positive tokens. The target distribution is
  the same as TNFF and TNRR, i.e., either the original probability distribution,
  if a positive weight is associated with the output token index, or a
  distribution where the output token is set to zero probability,
  if a negative weight is associated with the output token index.
  """

  def _get_desired_logits(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Get the desired logits to optimize towards.

    The desired distribution is either the original probability distribution,
    if no weight is associated with the output token index, or a probability
    distribution where the particular output token is near-zero probability,
    if a negative weight is associated with the output token index.

    We return logits instead of the normalized distribution since the function
    losses.cross_entropy_with_logits() expects logits and performs normalization
    within the function itself.

    Args:
      logits: model logits [batch, seq, vocab]
      target_tokens: target tokens, token idx only [batch, seq]
      weights: weights associated with the target tokens [batch, seq]
      orig_logits: logits from the original model before any training [batch,
        seq, vocab]

    Returns:
      A [batch, seq, vocab] array of the desired logits.
    """
    # target_tokens is [batch, seq]. targets is [batch, seq, num_classes]
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0
    )

    # zero out the probability at a target token if it is assigned a -1 weight
    # return original_dist if the token is assigned a 1 weight
    # since we're working with logits, the logits become:
    # original logits if the weight is 1
    # -infty at the target token and original logits elsewhere if weight is -1
    large_number = 1e9
    desired_logits = jnp.where(
        jnp.expand_dims(weights, -1) > 0,
        orig_logits,
        orig_logits + targets * jnp.expand_dims(weights, -1) * large_number,
    )

    # convert logits to a smoothed version so the resulting desired dist does
    # not have zeros and avoid problems when we take logs
    desired_dist = jax.nn.softmax(desired_logits)
    desired_dist_smoothed_unnorm = desired_dist + 1e-6
    desired_dist_smoothed = desired_dist_smoothed_unnorm / jnp.sum(
        desired_dist_smoothed_unnorm, axis=-1, keepdims=True
    )
    desired_logits_smoothed = jnp.log(desired_dist_smoothed)
    return desired_logits_smoothed

  def _get_token_losses(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute token-level losses."""
    desired_logits = self._get_desired_logits(
        logits, target_tokens, weights, orig_logits
    )
    rkl_mask = jnp.where(weights == -1, 1, 0)
    fkl_mask = jnp.where(weights == 1, 1, 0)
    ce_token_scores = (
        losses.cross_entropy_with_logits(
            desired_logits, jax.nn.softmax(logits, axis=-1), z_loss=0.0
        )[0]
        - losses.cross_entropy_with_logits(
            logits, jax.nn.softmax(logits, axis=-1), z_loss=0.0
        )[0]
    ) * rkl_mask * self.alpha + (
        losses.cross_entropy_with_logits(
            logits, jax.nn.softmax(desired_logits, axis=-1), z_loss=0.0
        )[0]
    ) * fkl_mask
    return ce_token_scores  # [batch, seq]


class TNFLLModel(SelfDistillationModel, abc.ABC):
  """Targeted negative training with forward KL and sample log likelihood terms.

  For negative tokens, we minimize the forward KL divergence between the desired
  distribution and the current model distribution (via minimizing the cross
  entropy between the target and model distributions). For positive tokens, we
  simply maximize their likelihood (i.e., minimize the cross entropy between the
  sample as a one-hot encoded vector and the model distribution), not taking
  into account the original model output distribution.
  """

  def _get_desired_dist(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Return the desired conditional distributions to optimize towards.

    The desired distribution is either the original probability distribution,
    if a positive weight is associated with the output token index, or a
    distribution where the output token is set to zero probability,
    if a negative weight is associated with the output token index.

    Args:
      logits: model logits [batch, seq, vocab]
      target_tokens: target tokens, token idx only [batch, seq]
      weights: weights associated with the target tokens [batch, seq]
      orig_logits: logits from the original model before any training [batch,
        seq, vocab]

    Returns:
      A [batch, seq, vocab] array of the desired distribution.
    """
    # target_tokens is [batch, seq]. targets is [batch, seq, num_classes]
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0
    )

    original_dist = jax.nn.softmax(orig_logits, axis=-1)
    # zero out the probability at a target token if it is assigned a -1 weight
    # return original_dist otherwise
    desired_dist_unnorm = jnp.where(
        jnp.expand_dims(weights, -1) > 0,
        targets,
        original_dist + targets * jnp.expand_dims(weights, -1) * original_dist,
    )

    desired_dist = desired_dist_unnorm / desired_dist_unnorm.sum(
        -1, keepdims=True
    )
    return desired_dist

  def _get_token_losses(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute token-level losses."""
    desired_dist = self._get_desired_dist(
        logits, target_tokens, weights, orig_logits
    )
    neg_mask = jnp.where(weights < 0, 1, 0)
    pos_mask = jnp.where(weights > 0, 1, 0)
    ce_token_scores = losses.cross_entropy_with_logits(
        logits, desired_dist, z_loss=0.0
    )[0] * (neg_mask * self.alpha + pos_mask)
    return ce_token_scores


class TNRLLModel(SelfDistillationModel, abc.ABC):
  """Targeted negative training with reverse KL and sample log likelihood terms.

  For negative tokens, we minimize the reverse KL divergence between the desired
  distribution and the current model distribution. For positive tokens, we
  simply maximize their likelihood (i.e., minimize the cross entropy between the
  sample as a one-hot encoded vector and the model distribution), not taking
  into account the original model output distribution.
  """

  def _get_desired_logits(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    """Get the desired logits to optimize towards.

    The desired distribution is either the one-hot encoded output token,
    if no weight is associated with the output token index, or a probability
    distribution where the particular output token is near-zero probability,
    if a negative weight is associated with the output token index.

    We return logits instead of the normalized distribution since the function
    losses.cross_entropy_with_logits() expects logits and performs normalization
    within the function itself.

    Args:
      logits: model logits [batch, seq, vocab]
      target_tokens: target tokens, token idx only [batch, seq]
      weights: weights associated with the target tokens [batch, seq]
      orig_logits: logits from the original model before any training [batch,
        seq, vocab]

    Returns:
      A [batch, seq, vocab] array of the desired logits.
    """
    # target_tokens is [batch, seq]. targets is [batch, seq, num_classes]
    targets = common_utils.onehot(
        target_tokens, logits.shape[-1], on_value=1, off_value=0
    )

    # zero out the probability at a target token if it is assigned a -1 weight
    # return the one-hot-encoded target if the token is assigned a 1 weight
    # since we're working with logits, the logits become:
    # infty at the token and zero elsewhere if the token is assigned a 1 weight
    # -infty at the target token and original logits elsewhere if weight is -1
    large_number = 1e9
    desired_logits = jnp.where(
        jnp.expand_dims(weights, -1) > 0,
        targets * large_number,
        orig_logits + targets * jnp.expand_dims(weights, -1) * large_number,
    )

    # convert logits to a smoothed version so the resulting desired dist does
    # not have zeros and avoid problems when we take logs
    desired_dist = jax.nn.softmax(desired_logits)
    desired_dist_smoothed_unnorm = desired_dist + 1e-6
    desired_dist_smoothed = desired_dist_smoothed_unnorm / jnp.sum(
        desired_dist_smoothed_unnorm, axis=-1, keepdims=True
    )
    desired_logits_smoothed = jnp.log(desired_dist_smoothed)
    return desired_logits_smoothed

  def _get_token_losses(
      self,
      logits: jnp.ndarray,
      target_tokens: jnp.ndarray,
      weights: jnp.ndarray,
      orig_logits: jnp.ndarray,
  ) -> jnp.ndarray:
    desired_logits = self._get_desired_logits(
        logits, target_tokens, weights, orig_logits
    )
    rkl_mask = jnp.where(weights == -1, 1, 0)
    fkl_mask = jnp.where(weights == 1, 1, 0)
    ce_token_scores = (
        losses.cross_entropy_with_logits(
            desired_logits, jax.nn.softmax(logits, axis=-1), z_loss=0.0
        )[0]
        - losses.cross_entropy_with_logits(
            logits, jax.nn.softmax(logits, axis=-1), z_loss=0.0
        )[0]
    ) * rkl_mask * self.alpha + (
        losses.cross_entropy_with_logits(
            logits, jax.nn.softmax(desired_logits, axis=-1), z_loss=0.0
        )[0]
    ) * fkl_mask
    return ce_token_scores  # [batch, seq]


@gin.configurable(module='models')
class EncoderDecoderModelNL(NLModel, EncoderDecoderModel):
  """EncoderDecoderModel with negative likelihood loss and score.

  Loss is negative log likelihood on positive token indices and the negation
  (i.e., log likelihood) on negative token indices.
  """

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTree] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    return super()._compute_logits(
        params, batch, dropout_rng, mutable, other_variables
    )


@gin.configurable(module='models')
class EncoderDecoderModelUL(ULModel, EncoderDecoderModel):
  """EncoderDecoderModel with unlikelihood loss and score.

  Loss is negative log likelihood on positive token indices and unlikelihood
  on negative token indices.
  """

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTree] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    return super()._compute_logits(
        params, batch, dropout_rng, mutable, other_variables
    )


@gin.configurable(module='models')
class EncoderDecoderModelTNFF(TNFFModel, EncoderDecoderModel):
  """EncoderDecoderModel with targeted negative loss and score (ff-variant).

  Loss is forward KL divergence, i.e., KL(p_d || p_m), where p_d is the desired
  distribution (the original model distribution for positive indices, the
  original model distribution with negatives pushed down for negative indices).
  """

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTree] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    return super()._compute_logits(
        params, batch, dropout_rng, mutable, other_variables
    )


@gin.configurable(module='models')
class EncoderDecoderModelTNRR(TNRRModel, EncoderDecoderModel):
  """EncoderDecoderModel with targeted negative loss and score (rr-variant).

  Loss is reverse KL divergence, i.e., KL(p_m || p_d), where p_d is the desired
  distribution (the original model distribution for positive indices, the
  original model distribution with negatives pushed down for negative indices).
  """

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTree] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    return super()._compute_logits(
        params, batch, dropout_rng, mutable, other_variables
    )


@gin.configurable(module='models')
class EncoderDecoderModelTNRF(TNRFModel, EncoderDecoderModel):
  """EncoderDecoderModel with targeted negative loss and score (rf-variant).

  Loss is reverse KL divergence, i.e., KL(p_m || p_d), for negative indices
  and forward KL divergence, i.e., KL(p_d || p_m), for positive indices.
  p_d is the desired distribution and p_m is the current model distribution.
  """

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTree] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    return super()._compute_logits(
        params, batch, dropout_rng, mutable, other_variables
    )


@gin.configurable(module='models')
class EncoderDecoderModelTNFLL(TNFLLModel, EncoderDecoderModel):
  """EncoderDecoderModel with targeted negative loss and score (fll-variant).

  Loss is forward KL divergence, i.e., KL(p_d || p_m), for negative indices
  and negative log-likelihood on the sample for positive indices.
  """

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTree] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    return super()._compute_logits(
        params, batch, dropout_rng, mutable, other_variables
    )


@gin.configurable(module='models')
class EncoderDecoderModelTNRLL(TNRLLModel, EncoderDecoderModel):
  """EncoderDecoderModel with targeted negative loss and score (rll-variant).

  Loss is reverse KL divergence, i.e., KL(p_d || p_m), for negative indices
  and negative log-likelihood on the sample for positive indices.
  """

  def _compute_logits(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTree] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    return super()._compute_logits(
        params, batch, dropout_rng, mutable, other_variables
    )
