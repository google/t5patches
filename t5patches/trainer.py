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

"""t5x trainer training with the self-distillation.

All the logic here is near-identical to that of the t5x trainer. The primary
difference is that the SelfDistillationTrainer keeps track of the original model
params at the start of training, and passes those parameters in as arguments to
calculate gradients.

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
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, TYPE_CHECKING, Tuple, Union

from absl import logging
import cached_property
import clu.data
import clu.metrics
import clu.values
from flax.core import FrozenDict
import jax.lax
import jax.numpy as jnp
import jax.random
import numpy as np
from t5patches import models
from t5x import metrics as metrics_lib
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x.trainer import apply_grads
from t5x.trainer import LearningRateCallable
from t5x.trainer import merge_metrics
from t5x.trainer import PartitionedEvalCallable
from t5x.trainer import PartitionedTrainCallable
from t5x.trainer import Trainer
from t5x.trainer import WeightMetricsComputer

Array = Union[np.ndarray, jnp.ndarray]
BatchSpec = Mapping[str, jax.ShapeDtypeStruct]
BatchType = Mapping[str, np.ndarray]
FlaxMutables = FrozenDict
Rng = jnp.ndarray
MetricMapType = MutableMapping[str, clu.metrics.Metric]
MetricMapSpec = Mapping[str, jax.ShapeDtypeStruct]
MetricValueMapType = Mapping[str, clu.values.Value]
ModelWeights = Any
MutableMetricMapType = Dict[str, clu.metrics.Metric]
PartitionSpec = partitioning.PartitionSpec

if TYPE_CHECKING:  # See b/163639353
  cached_property = property  # pylint: disable=invalid-name
else:
  cached_property = cached_property.cached_property


def eval_step(model: models.SelfDistillationEncoderDecoderModel,
              train_state: train_state_lib.TrainState, batch: jnp.ndarray,
              orig_train_state: train_state_lib.TrainState) -> MetricMapType:
  """Default evaluation step.

  Same as t5x eval_step, except the evaluation function also takes in original
  model params as input.

  Args:
    model: an instance of a SelfDistillationEncoderDecoderModel to evaluate.
    train_state: current train state (parameters and optimizer state).
    batch: batch of data.
    orig_train_state: original train state before training began.

  Returns:
    A Mapping from metric names to their Metric values.

  """
  _, metrics = model.eval_fn(train_state.params, batch, orig_train_state.params)  # pytype: disable=wrong-arg-types  # jax-ndarray
  return metrics


def train_with_lr(
    train_state: train_state_lib.TrainState,
    batch: BatchType,
    learning_rate: jnp.ndarray,
    dropout_rng: Rng,
    model: models.SelfDistillationEncoderDecoderModel,
    num_microbatches: Optional[int],
    weight_metrics_computer: Optional[WeightMetricsComputer] = None,
    data_partition_spec: PartitionSpec = PartitionSpec("data"),
    orig_train_state: Optional[train_state_lib.TrainState] = None
) -> Tuple[train_state_lib.TrainState, MetricMapType]:
  """Main training function with LR schedule.

  Same as t5 train_with_lr, except the function also takes in original
  model params as input.

  Args:
    train_state: current train state (parameters and optimizer state).
    batch: a batch of data.
    learning_rate: learning rate for the gradient step.
    dropout_rng: jax PRNGKey for dropout.
    model: an instance of a SelfDistillationEncoderDecoderModel to train.
    num_microbatches: the number of microbatches to use, or None for direct
      training.
    weight_metrics_computer: A WeightMetricsComputer instance, or None, to
      decide what metrics, if any, to log about weights and weight updates
      during training.
    data_partition_spec: the PartitionSpec to use for partitioning annotations
      on the batch.
    orig_train_state: the original train state before training began.

  Returns:
    Tuple of the new train_state and metrics

  """
  grad_accum, metrics, flax_mutables = (
      accumulate_grads_microbatched(
          model,
          train_state,
          batch,
          dropout_rng,
          num_microbatches,
          data_partition_spec,
          orig_train_state=orig_train_state))
  new_train_state, metrics = apply_grads(
      train_state,
      grad_accum,
      metrics,
      learning_rate,
      weight_metrics_computer,
      other_state_variables={"flax_mutables": flax_mutables}
      if flax_mutables else None)

  return new_train_state, metrics


def accumulate_grads_microbatched(
    model: models.SelfDistillationEncoderDecoderModel,
    train_state: train_state_lib.TrainState,
    batch: BatchType,
    dropout_rng: Rng,
    num_microbatches: Optional[int],
    data_partition_spec: PartitionSpec = PartitionSpec("data"),
    orig_train_state: Optional[train_state_lib.TrainState] = None,
) -> Tuple[train_state_lib.TrainState, MutableMetricMapType,
           Optional[FlaxMutables]]:
  """Implements optional microbatched gradient accumulation.

  Same logic as accumulate_grads_microbatched, except that the grad_fn
  can optionally take in original model params.

  Args:
    model: the instantiation of `SelfDistillationEncoderDecoderModel` to train.
    train_state: A train state with model parameters and optimizer state.
    batch: input batch consisting of either - simply-padded batched features
      'encoder_input_tokens', 'decoder_input_tokens' 'decoder_target_tokens'
      'decoder_loss_weights'- packed, batched features with additional
      "(encoder|decoder)_segment_id", "(encoder|decoder)_position"
    dropout_rng: jax PRNGKey for dropout.
    num_microbatches: the number of microbatches to use, or None for direct
      training.
    data_partition_spec: the PartitionSpec to use for partitioning annotations
      on the batch.
    orig_train_state: A train state with the original model parameters and
      optimizer state at the beginning of the training.

  Returns:
   Accumulated gradients and incremental metrics.
  """
  batch_size = next(iter(batch.values())).shape[0]

  grad_fn = jax.value_and_grad(model.loss_fn, has_aux=True)

  # We assume that the model loss_fn supports flax mutables if and only if
  # the train state includes non-empty flax mutables.
  # Note: Default t5x models don't support flax_mutables. One needs to subclass
  # them and return flax_mutables from `get_initial_variables` and `loss_fn`.

  initial_flax_mutables = train_state.flax_mutables if train_state.flax_mutables else None

  if num_microbatches is None or num_microbatches <= 1:

    if initial_flax_mutables is None:
      if orig_train_state is None:
        (_, metrics), grad_accum = grad_fn(train_state.params, batch,
                                           dropout_rng)
      else:
        (_, metrics), grad_accum = grad_fn(train_state.params, batch,
                                           dropout_rng, orig_train_state.params)
      flax_mutables = None
    else:
      if orig_train_state is None:
        (_, (metrics,
             flax_mutables)), grad_accum = grad_fn(train_state.params, batch,
                                                   dropout_rng,
                                                   initial_flax_mutables)
      else:
        (_, (metrics, flax_mutables)), grad_accum = grad_fn(
            train_state.params, batch, dropout_rng, orig_train_state.params,
            initial_flax_mutables)
  else:
    assert batch_size % num_microbatches == 0, (
        "Batch size isn't divided evenly by num_microbatches.")
    microbatch_size = batch_size // num_microbatches
    logging.info("using microbatches: %d microbatches, %d size",
                 num_microbatches, microbatch_size)

    def get_microbatch(batch: BatchType, idx: int) -> Mapping[str, jnp.ndarray]:
      """Fetch microbatch slice from possibly-packed input data."""
      offset = idx * microbatch_size
      length = microbatch_size
      starts = {k: [offset] + [0] * (b.ndim - 1) for k, b in batch.items()}
      limits = {k: [length] + list(b.shape[1:]) for k, b in batch.items()}
      return {
          k: jax.lax.dynamic_slice(b, starts[k], limits[k])
          for k, b in batch.items()
      }

    def metrics_and_grad(loop_cnt, dropout_rng, flax_mutables=None):
      dropout_rng, sub_dropout_rng = jax.random.split(dropout_rng)
      mbatch = get_microbatch(batch, loop_cnt)
      # We need to annotate the microbatch sharding as we would a batch.
      mbatch = jax.tree_util.tree_map(
          lambda x: partitioning.with_sharding_constraint(  # pylint: disable=g-long-lambda
              x, data_partition_spec
          ),
          mbatch,
      )
      if flax_mutables is None:
        (_, metrics), grad = grad_fn(train_state.params, mbatch,
                                     sub_dropout_rng)
      else:
        (_, (metrics, flax_mutables)), grad = grad_fn(train_state.params,
                                                      mbatch, sub_dropout_rng,
                                                      flax_mutables)
      return metrics, grad, flax_mutables

    def per_microbatch_train_step(
        loop_cnt: int, state: Tuple[jnp.ndarray, jnp.ndarray,
                                    Mapping[str, jnp.ndarray],
                                    Optional[FlaxMutables]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray],
               Optional[FlaxMutables]]:
      (dropout_rng, grad_accum, prev_metrics, flax_mutables) = state
      metrics, grad, flax_mutables = metrics_and_grad(loop_cnt, dropout_rng,
                                                      flax_mutables)

      grad_accum = jax.tree_util.tree_map(jnp.add, grad_accum, grad)
      metrics = jax.lax.cond(loop_cnt == 0, lambda _: metrics,
                             lambda _: merge_metrics(prev_metrics, metrics),
                             None)
      return dropout_rng, grad_accum, metrics, flax_mutables

    # Initialize gradient accumulation loop state.
    accum_dtype = jnp.float32
    grad_accum_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, accum_dtype), train_state.params
    )
    initial_metrics_shape, _, _ = jax.eval_shape(
        metrics_and_grad, loop_cnt=0, dropout_rng=dropout_rng)

    initial_metrics = {
        k: metrics_lib.shape_obj_to_defined_obj(v)
        for k, v in initial_metrics_shape.items()
    }
    loop_init = (dropout_rng, grad_accum_init, initial_metrics,
                 initial_flax_mutables)
    new_dropout_rng, grad_accum, metrics, flax_mutables = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, loop_init)

    del new_dropout_rng

  return grad_accum, metrics, flax_mutables


class SelfDistillationTrainer(Trainer):
  """Training loop with optional microbatches."""

  def __init__(self,
               model: models.SelfDistillationEncoderDecoderModel,
               train_state: train_state_lib.TrainState,
               partitioner: partitioning.BasePartitioner,
               eval_names: Sequence[str],
               summary_dir: Optional[str],
               train_state_axes: Any,
               rng: Rng,
               learning_rate_fn: LearningRateCallable,
               num_microbatches: Optional[int],
               weight_metrics_computer: Optional[WeightMetricsComputer] = None,
               orig_train_state: Optional[train_state_lib.TrainState] = None):
    """Trainer for training with self-distillation.

    Behaves exactly like the t5x Trainer, except that this Trainer also keeps
    track of the original model params at the start of training and passes
    those parameters in as arguments to calculate gradients. Trainer to use
    for targeted negative (TN) training.

    Args:
      model: instantiation of `SelfDistillationEncoderDecoderModel` to train.
      train_state: a train state with parameters and optimizer state.
      partitioner: the partitioner to use.
      eval_names: names of evaluation datasets, which must match the keys of the
        mapping passed to `eval`.
      summary_dir: optional directory to write TensorBoard metrics to.
      train_state_axes: partitioning info for the optimizer to be used.
      rng: jax PRNGKey seed for random operations, to be combined with step
        number for a deterministic RNG.
      learning_rate_fn: returns the learning rate given the current step.
      num_microbatches: the number of microbatches to use, or None for direct
        training.
      weight_metrics_computer: A WeightMetricsComputer instance, or None, to
        decide what metrics, if any, to log about weights and weight updates
        during training.
      orig_train_state: the original train state of the model to compare
        against. If None is passed, orig_train_state is set to train_state, e.g.
        the model train state at the beginning of training when the trainer is
        initialized.
    """
    if orig_train_state is None:
      self.orig_train_state = train_state
    else:
      self.orig_train_state = orig_train_state

    super().__init__(
        model=model,
        train_state=train_state,
        partitioner=partitioner,
        eval_names=eval_names,
        summary_dir=summary_dir,
        train_state_axes=train_state_axes,
        rng=rng,
        learning_rate_fn=learning_rate_fn,
        num_microbatches=num_microbatches,
        weight_metrics_computer=weight_metrics_computer)

    # assignment happens in base class, but here we explicitly type to a
    # narrower model class
    self._model: models.SelfDistillationEncoderDecoderModel = model

  @cached_property
  def _partitioned_train_step(self) -> PartitionedTrainCallable:

    def train_step(
        train_state: train_state_lib.TrainState,
        batch: BatchType) -> Tuple[train_state_lib.TrainState, MetricMapType]:
      return train_with_lr(
          train_state,
          batch,
          learning_rate=self._learning_rate_fn(train_state.step),
          dropout_rng=self._get_step_rng(train_state.step),  # pytype: disable=wrong-arg-types  # jax-ndarray
          model=self._model,
          num_microbatches=self._num_microbatches,
          weight_metrics_computer=self._weight_metrics_computer,
          data_partition_spec=self._partitioner.data_partition_spec,
          orig_train_state=self.orig_train_state,
      )

    return self._partitioner.partition(
        train_step,
        in_axis_resources=(
            self._train_state_axes,
            self._partitioner.data_partition_spec,
        ),
        out_axis_resources=(self._train_state_axes, None),
        donate_argnums=(0,),
    )

  @cached_property
  def _partitioned_eval_step(self) -> PartitionedEvalCallable:
    return self._partitioner.partition(
        lambda *args, **kwargs: eval_step(  # pylint: disable=g-long-lambda
            self._model,
            *args,
            orig_train_state=self.orig_train_state,
            **kwargs),
        in_axis_resources=(self._train_state_axes,
                           self._partitioner.data_partition_spec),
        out_axis_resources=None)
