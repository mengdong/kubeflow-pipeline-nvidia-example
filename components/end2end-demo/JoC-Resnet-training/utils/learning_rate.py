#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

import tensorflow as tf

__all__ = ['learning_rate_scheduler']


def learning_rate_scheduler(learning_rate_init, global_step, batch_size, num_batches_per_epoch, num_gpus):

    def get_scaled_base_learning_rate():
        """Calculates base learning rate for creating lr schedule.
        In replicated mode, gradients are summed rather than averaged which, with
        the sgd and momentum optimizers, increases the effective learning rate by
        lr * num_gpus. Dividing the base lr by num_gpus negates the increase.
        Args:
          batch_size: Total batch-size.
        Returns:
          Base learning rate to use to create lr schedule.
        """

        base_lr = learning_rate_init * num_gpus

        # Starting LR = 0.1 with BS = 256, else linearly scale
        # return base_lr * (batch_size / 256.0)
        return base_lr

    rescaled_lr = get_scaled_base_learning_rate()

    boundaries = [int(num_batches_per_epoch * x) for x in [20, 40, 60, 80]]

    values = [1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    values = [rescaled_lr * v for v in values]

    lr = tf.train.piecewise_constant(global_step, boundaries, values)

    # First 1 Epochs are warmup
    warmup_steps = int(num_batches_per_epoch * 5)

    warmup_lr = (rescaled_lr * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

    return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
