# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================
"""Training script for GCN algorithm"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import graphlearn as gl
import tensorflow as tf

from gcn import GCN


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('task_index', 0, 'Task index')
flags.DEFINE_string('job_name', '', 'Worker or server')
flags.DEFINE_string('worker_hosts', '', 'Worker hosts string')
flags.DEFINE_string('ps_hosts', '', 'PS hosts')
flags.DEFINE_string('tracker', '/filepool/test-tracker/', 'tracker dir')
flags.DEFINE_string('dataset', 'cora', 'Dataset to train on ')



def load_graph(config):
  """
  load graph from source, decode
  :return:
  """
  dataset_folder = config['dataset_folder']
  node_type = config['node_type']
  edge_type = config['edge_type']
  g = gl.Graph()\
        .node(dataset_folder + "node_table", node_type=node_type,
              decoder=gl.Decoder(labeled=True,
                                 attr_types=["float"] * (config['features_num']),
                                 attr_delimiter=":"))\
        .edge(dataset_folder + "edge_table_with_self_loop", edge_type=(node_type, node_type, edge_type),
              decoder=gl.Decoder(weighted=True), directed=True)\
        .node(dataset_folder + "train_table", node_type="train",
              decoder=gl.Decoder(weighted=True))\
        .node(dataset_folder + "val_table", node_type="val",
              decoder=gl.Decoder(weighted=True))\
        .node(dataset_folder + "test_table", node_type="test",
              decoder=gl.Decoder(weighted=True))
  return g


def train(config, graph):
  def model_fn():
    return GCN(graph,
               config['class_num'],
               config['features_num'],
               config['batch_size'],
               val_batch_size=config['val_batch_size'],
               test_batch_size=config['test_batch_size'],
               categorical_attrs_desc=config['categorical_attrs_desc'],
               hidden_dim=config['hidden_dim'],
               in_drop_rate=config['in_drop_rate'],
               hops_num=config['hops_num'],
               neighs_num=config['neighs_num'],
               full_graph_mode = config['full_graph_mode'])

  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
  trainer = gl.DistTFTrainer(model_fn,
                             cluster_spec=cluster,
                             task_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             epoch=config['epoch'],
                             optimizer=gl.get_tf_optimizer(
                                 config['learning_algo'],
                                 config['learning_rate'],
                                 config['weight_decay']),
                             val_frequency=1)

  if FLAGS.job_name == 'worker':
    trainer.train_and_evaluate()
  else:
    trainer.join()


def convert_dict(d):
  for k in d.keys():
    if isinstance(d[k], unicode):
      d[k] = d[k].encode('utf-8')

  return d

def main():
  config = json.load(open('configs/' + FLAGS.dataset + '_config.json', 'r'))
  config = convert_dict(config)

  g = load_graph(config)
  if FLAGS.job_name == 'worker':
    g.init(tracker=FLAGS.tracker, task_index=FLAGS.task_index, hosts=FLAGS.worker_hosts)
  train(config, g)

  g.close()


if __name__ == "__main__":
  main()
