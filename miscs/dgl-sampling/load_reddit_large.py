from __future__ import absolute_import

import scipy.sparse as sp
import numpy as np
import os

from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs, deprecate_property
from dgl import backend as F
from dgl.convert import from_scipy


class RedditLargeDataset(DGLDataset):
    r""" Reddit large dataset

    Statistics

    - Nodes: 1,019,186
    - Edges: 1,315,562,866
    - Node feature size: 301
    - Number of training samples: 672,663
    - Number of validation samples: 101,918
    - Number of test samples: 244,605

    Attributes
    ----------
    num_classes : int
        Number of classes for each node
    graph : :class:`dgl.DGLGraph`
        Graph of the dataset
    num_labels : int
        Number of classes for each node
    train_mask: numpy.ndarray
        Mask of training nodes
    val_mask: numpy.ndarray
        Mask of validation nodes
    test_mask: numpy.ndarray
        Mask of test nodes
    features : Tensor
        Node features
    labels :  Tensor
        Node labels
    """
    def __init__(self, raw_dir='/home/yifan/dataset/reddit_large', force_reload=False, verbose=False):
        super(RedditLargeDataset, self).__init__(name='reddit_large',
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
        # graph
        coo_adj = sp.load_npz(os.path.join(self._raw_dir, "reddit_large_graph.npz"))
        self._graph = from_scipy(coo_adj)
        # features and labels
        reddit_data = np.load(os.path.join(self._raw_dir, "reddit_large_data.npz"))
        features = reddit_data["feature"]
        labels = reddit_data["label"]
        # tarin/val/test indices
        node_types = reddit_data["node_types"]
        train_mask = (node_types == 1)
        val_mask = (node_types == 2)
        test_mask = (node_types == 3)
        self._graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
        self._graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
        self._graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
        self._graph.ndata['feat'] = F.tensor(features, dtype=F.data_type_dict['float32'])
        self._graph.ndata['label'] = F.tensor(labels, dtype=F.data_type_dict['int64'])
        self._print_info()

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
        self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
        self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())
        self._print_info()

    def _print_info(self):
        print('Finished data loading.')
        print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self._graph.number_of_edges()))
        print('  NumFeats: {}'.format(self._graph.ndata['feat'].shape[1]))
        print('  NumClasses: {}'.format(self.num_classes))
        print('  NumTrainingSamples: {}'.format(F.nonzero_1d(self._graph.ndata['train_mask']).shape[0]))
        print('  NumValidationSamples: {}'.format(F.nonzero_1d(self._graph.ndata['val_mask']).shape[0]))
        print('  NumTestSamples: {}'.format(F.nonzero_1d(self._graph.ndata['test_mask']).shape[0]))

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 50

    @property
    def num_labels(self):
        deprecate_property('dataset.num_labels', 'dataset.num_classes')
        return self.num_classes

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'dataset[0]')
        return self._graph

    @property
    def train_mask(self):
        deprecate_property('dataset.train_mask', 'graph.ndata[\'train_mask\']')
        return F.asnumpy(self._graph.ndata['train_mask'])

    @property
    def val_mask(self):
        deprecate_property('dataset.val_mask', 'graph.ndata[\'val_mask\']')
        return F.asnumpy(self._graph.ndata['val_mask'])

    @property
    def test_mask(self):
        deprecate_property('dataset.test_mask', 'graph.ndata[\'test_mask\']')
        return F.asnumpy(self._graph.ndata['test_mask'])

    @property
    def features(self):
        deprecate_property('dataset.features', 'graph.ndata[\'feat\']')
        return self._graph.ndata['feat']

    @property
    def labels(self):
        deprecate_property('dataset.labels', 'graph.ndata[\'label\']')
        return self._graph.ndata['label']

    def __getitem__(self, idx):
        r""" Get graph by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node labels, node features and splitting masks:

            - ``ndata['label']``: node label
            - ``ndata['feat']``: node feature
            - ``ndata['train_mask']``： mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']:`` mask for test node set
        """
        assert idx == 0, "Reddit Dataset only has one graph"
        return self._graph

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1

