import random

import numpy as np

from operator import itemgetter
from collections import defaultdict

from .base import KnowledgeGraph
from .heap import Heap


class Summary(KnowledgeGraph):

    def __init__(self, KG):
        """
        :param KG: KnowledgeGraph
        """
        super().__init__()
        self.parent_ = KG

    def parent(self):
        return self.parent_

    def marginal_value(self, triple):
        """
        :param triple: (e1, r, e2) triple
        :return marginal_value: total marginal value of adding triple to S
        """
        total = 0
        e1, r, e2 = triple

        if not self.has_entity(e1):
            total += self.parent().entity_value(e1)
        if not self.has_entity(e2):
            total += self.parent().entity_value(e2)
        if not self.has_triple(triple):
            total += self.parent().triple_value(triple)
        return total

    def fill(self, triples, k):
        """
        :param triples: triples to add to summary
        :param k: limit
        """
        for triple in triples:
            if self.number_of_triples() >= k:
                return
            self.add_triple(triple)


class SummaryMethod(object):
    """Stores a summarization function and associated metadata."""

    def __init__(self, fn, name, **kwargs):
        """
        :param fn: summarization function to call
        :param name: pretty-printed name of this function
        :param kwargs: optional keyword arguments for fn
        """
        self.fn_ = fn
        self.name_ = name
        self.kwargs_ = kwargs

    def name(self):
        return self.name_

    def kwargs(self):
        return self.kwargs_

    def __call__(self, KG, K, query_log):
        """
        :param KG: KnowledgeGraph
        :param K: summary constraint
        :param query_log: query log
        :return results: results of function call
        """
        return self.fn_(KG, K, query_log, **self.kwargs_)


def GLIMPSE(KG, K, query_log, epsilon=1e-3, power=1):
    """
    :param KG: KnowledgeGraph to summarize
    :param K: number of triples in summary
    :param query_log: user queries
    :param epsilon: float in (0, 1] or None, epsilon-from-optimal factor
    :param power: number of terms in Taylor expansion
    :return S: Summary
    """
    # Estimate user preferences over KG
    KG.model_user_pref(query_log, power=power)

    # Greedily select top-k triples for summary S
    heap = Heap(KG)
    S = Summary(KG)

    if len(heap) <= K:
        S.fill(heap.triples(), K)
    else:
        heap.update(S, len(heap)) # update all marginals
        sample_size = len(heap) if epsilon is None else \
                int(len(heap) / K * np.log(1 / epsilon))

        while len(heap) and S.number_of_triples() < K:
            triple = heap.pop()
            S.add_triple(triple)
            heap.update(S, sample_size)

    S.fill(KG.triples(), K)
    return S

