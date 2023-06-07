import numpy as np
import dgl, torch, math

class GraphExample():

    pass

class BatchedGraph():

    pass

class GraphFactory():

    def __init__(self, method='rgatsql', relation_vocab=None):
        super(GraphFactory, self).__init__()
        self.method = eval('self.' + method)
        self.batch_method = eval('self.batch_' + method)
        self.relation_vocab = relation_vocab

    def graph_construction(self, ex: dict, db: dict):
        return self.method(ex, db)

    def rgatsql(self, ex, db):
        graph = GraphExample()

        global_edges = ex['graph'].global_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], global_edges))
        graph.global_edges = torch.tensor(rel_ids, dtype=torch.long)


        if 'global_edges2' in ex['graph'].__dict__.keys():
            global_edges2 = ex['graph'].global_edges2
            rel_ids = list(map(lambda r: self.relation_vocab[r[2]], global_edges2))
            graph.global_edges2 = torch.tensor(rel_ids, dtype=torch.long)

        global_edge_map = {}
        for index in range(len(global_edges)):
            edge = global_edges[index]
            global_edge_map[(edge[0], edge[1])] = index
        graph.global_edge_map = global_edge_map

        graph.global_g = ex['graph'].global_g
        graph.gp = ex['graph'].gp
        graph.question_mask = torch.tensor(ex['graph'].question_mask, dtype=torch.bool)
        graph.schema_mask = torch.tensor(ex['graph'].schema_mask, dtype=torch.bool)
        graph.node_label = torch.tensor(ex['graph'].node_label, dtype=torch.float)
        global_enum = graph.global_edges.size(0)

        return graph

    def batch_graphs(self, ex_list, device, train=True, **kwargs):

        return self.batch_method(ex_list, device, train=train, **kwargs)

    def batch_rgatsql(self, ex_list, device, train=True, **kwargs):

        graph_list = ex_list
        bg = BatchedGraph()

        bg.global_g = dgl.batch([ex.global_g for ex in graph_list]).to(device)
        bg.global_edges = torch.cat([ex.global_edges for ex in graph_list], dim=0).to(device)

        if 'global_edges2' in graph_list[0].__dict__.keys():
            bg.global_edges2 = torch.cat([ex.global_edges2 for ex in graph_list], dim=0).to(device)

        if train:
            bg.question_mask = torch.cat([ex.question_mask for ex in graph_list], dim=0).to(device)
            bg.schema_mask = torch.cat([ex.schema_mask for ex in graph_list], dim=0).to(device)
            smoothing = kwargs.pop('smoothing', 0.0)
            node_label = torch.cat([ex.node_label for ex in graph_list], dim=0)
            node_label = node_label.masked_fill_(~ node_label.bool(), 2 * smoothing) - smoothing
            bg.node_label = node_label.to(device)
            bg.gp = dgl.batch([ex.gp for ex in graph_list]).to(device)
        return bg
