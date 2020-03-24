import heapq

import numpy as np


class Heap(object):

    class Triple(object):
        """For use inside the Heap class only"""

        def __init__(self, triple, value):
            """
            :param triple: (e1, r, e2)
            :param value: marginal value of this triple
            """
            self.triple_, self.value_ = triple, value

        def triple(self):
            return self.triple_

        def _marginal_value(self):
            return self.value_

        def __eq__(self, other):
            """
            :param other: Triple
            :return: whether triples have same entities/relation
            """
            return self.triple() == other.triple()

        def __lt__(self, other):
            """Make Triples sortable"""
            return self._marginal_value() > other._marginal_value()

    def __init__(self, KG):
        """
        :param KG: KnowledgeGraph
        """
        self.heap_ = []

        for triple in KG.triples():
            e1, r, e2 = triple
            total = KG.entity_value(e1) + \
                    KG.entity_value(e2) + \
                    KG.triple_value(triple)

            if total > 0:
                self.heap_.append(Heap.Triple(triple, total))

    def __len__(self):
        return len(self.heap_)

    def triples(self):
        return [triple.triple() for triple in self.heap_]

    def pop(self):
        if not len(self.heap_):
            raise ValueError('Cannot pop from an empty heap')
        return self.heap_.pop().triple()

    def _update_marginal(self, S, item):
        """
        :param S: Summary
        :param item: Triple
        """
        item.value_ = S.marginal_value(item.triple())

    def _lazy_greedy(self, S, triples):
        """
        :param S: Summary
        :param triples: triples to add to heap
        :return top, triple: old + new tops of the heap
        """
        heapq.heapify(triples)
        top = triples[0]
        self._update_marginal(S, top)
        heapq.heapreplace(triples, top)
        return top, triples[0]

    def _move_to_top(self, argmax):
        self.heap_[-1], self.heap_[argmax] = self.heap_[argmax], self.heap_[-1]

    def _triples_at_index(self, indices):
        """
        :param indices: indices to get triples at
        :return triples: triples at specified indices
        """
        triples = []
        for i in indices:
            triple = self.heap_[i]
            triple.index_ = i
            triples.append(triple)
        return triples

    def update(self, S, sample_size):
        """
        :param S: Summary
        :param sample_size: size of sample for "lazy lazy greedy"
        """
        n = len(self.heap_)
        sample_size = min(n, sample_size)
        if n <= 1 or not sample_size:
            return

        # Sample according to "lazy lazy greedy"
        indices = np.random.randint(0, n, size=sample_size) if sample_size < n else np.arange(n)
        triples = self._triples_at_index(indices)
        before, after = self._lazy_greedy(S, triples)
        if before == after:
            self._move_to_top(after.index_)
            return

        # If lazy fails, update marginals of sampled set
        top, argmax = self.heap_[0], 0
        for i in indices:
            item = self.heap_[i]
            e1, r, e2 = item.triple()

            if S.has_entity(e1) or S.has_entity(e2):
                self._update_marginal(S, item)
            if top._marginal_value() < item._marginal_value():
                top, argmax = item, i

        self._move_to_top(argmax)
