# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function

from absl import app
from absl import flags as absl_flags
import tensorflow as tf

import benchmark_cnn
import cnn_util
import flags
from cnn_util import log_fn
FLAGS = tf.flags.FLAGS



flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)


def main(positional_arguments,num_gpus,batch_size,model,variable_update):
  # Command-line arguments like '--distortions False' are equivalent to
  # '--distortions=True False', where False is a positional argument. To prevent
  # this from silently running with distortions, we do not allow positional
  # arguments.
  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  params = benchmark_cnn.make_params_from_flags()
  params = benchmark_cnn.setup(params)
  bench = benchmark_cnn.BenchmarkCNN(params)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))
  
class GpuParamServerDeviceSetter(object):
  """Used with tf.device() to place variables on the least loaded GPU.

    A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
    'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
    placed on the least loaded gpu. All other Ops, which will be the computation
    Ops, will be placed on the worker_device.
  """

  def __init__(self, worker_device, ps_devices):
    """Initializer for GpuParamServerDeviceSetter.
    Args:
      worker_device: the device to use for computation Ops.
      ps_devices: a list of devices to use for Variable Ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device

    # Gets the least loaded ps_device
    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name

def _create_device_setter(is_cpu_ps, worker, num_gpus):
  """Create device setter object."""
  if is_cpu_ps:
    # tf.train.replica_device_setter supports placing variables on the CPU, all
    # on one GPU, or on ps_servers defined in a cluster_spec.
    return tf.train.replica_device_setter(
        worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
  else:
    gpus = ['/gpu:%d' % i for i in range(num_gpus)]
    return ParamServerDeviceSetter(worker, gpus)

# The method below is a modified snippet from the full example.
def _resnet_model_fn():
    # When set to False, variables are placed on the least loaded GPU. If set
    # to True, the variables will be placed on the CPU.
    is_cpu_ps = False

    # Loops over the number of GPUs and creates a copy ("tower") of the model on
    # each GPU.
    for i in range(num_gpus):
      worker = '/gpu:%d' % i
      # Creates a device setter used to determine where Ops are to be placed.
      device_setter = _create_device_setter(is_cpu_ps, worker, FLAGS.num_gpus)
      # Creates variables on the first loop.  On subsequent loops reuse is set
      # to True, which results in the "towers" sharing variables.
      with tf.variable_scope('resnet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          # tf.device calls the device_setter for each Op that is created.
          # device_setter returns the device the Op is to be placed on.
          with tf.device(device_setter):
            bench.print_info()
            bench.run()
            # Creates the "tower".
#            _tower_fn(is_training, weight_decay, tower_features[i],
#                      tower_labels[i], tower_losses, tower_gradvars,
#                      tower_preds, False)


#  bench.print_info()
#  bench.run()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--variable_update',
      type=str,
  parser.add_argument(
      '--num_gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
      '--model',
      type=str,
      default='resnet152',
      help='The number of steps to use for training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size for training.')
 
  args = parser.parse_args()

  if args.num_gpus > 0:
    assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
    raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')
  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid --num-layers parameter.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
    raise ValueError('--train-batch-size must be multiple of --num-gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
    raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

  app.run(main(**vars(args)))

#  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
