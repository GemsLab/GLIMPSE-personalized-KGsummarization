import os
import gzip
import json
import re

import numpy as np

from collections import defaultdict
from scipy.sparse import csr_matrix

from .algorithms import query_vector, random_walk_with_restart

# TODO: Replace these data directories with your own paths
FREEBASE_DATA_DIR = '/x/tsafavi/data/WebQSDP/data/'
YAGO_DATA_DIR = '/x/tsafavi/data/yago3/'
DBPEDIA_DATA_DIR = '/x/tsafavi/data/dbpedia3.5.1/'


class KnowledgeGraph(object):

    def __init__(self):
        """A KG is a set of entities E, a set of relationships R,
        and a set of triples E x R x E."""
        self.entities_ = set()
        self.relationships_ = set()
        self.triples_ = {}
        self.number_of_triples_ = 0

        # Map entities to numeric IDs
        self.eid_ = 0
        self.entity_id_ = {}
        self.id_entity_ = {}

        self.name_ = None

    def name(self):
        return self.name_

    def entities(self):
        """
        :return entities: all entities in the KG
        """
        return self.entities_

    def relationships(self):
        """
        :return relationships: all relations in the KG
        """
        return self.relationships_

    def triples(self):
        """
        :return triples: {(e1, r, e2) triples}

        Note that this method is linear in the number of triples
        in the KG because it has to create a flat set of triples.
        """
        triples = set()
        for e1 in self.triples_:
            for r in self.triples_[e1]:
                for e2 in self.triples_[e1][r]:
                    triples.add((e1, r, e2))
        return triples

    def number_of_entities(self):
        """
        :return n_entities: number of entities in the KG
        """
        return len(self.entities_)

    def number_of_relationships(self):
        """
        :return n_relations: number of relations in the KG
        """
        return len(self.relationships_)

    def number_of_triples(self):
        """
        :return n_triples: number of triples in the KG
        """
        return self.number_of_triples_

    def has_entity(self, entity):
        """
        :param entity: str
        :return has_entity: True if KG contains this entity
        """
        return entity in self.entities_

    def has_relationship(self, relationship):
        """
        :param relationship: str
        :return has_relationship: True if KG contains this relationship
        """
        return relationship in self.relationships_

    def __getitem__(self, entity):
        """
        :param entity: str
        :return d: dict of set of {relation : entities}
        """
        return self.triples_[entity]

    def __contains__(self, entity):
        """
        :param entity: str
        :return has_entity: True if KG contains this head entity
        """
        return entity in self.triples_

    def has_triple(self, triple):
        """
        :param triple: (e1, r, e2)
        :return has_triple: True if KG contains this triple
        """
        e1, r, e2 = triple
        return e1 in self.triples_ and r in self.triples_[e1] and e2 in self.triples_[e1][r]

    def add_triple(self, triple):
        """
        :param triple: (e1, r, e2) triple
        """
        e1, r, e2 = triple
        if not self.has_triple(triple):
            self.number_of_triples_ += 1
            self.relationships_.add(r)

            # Record new entities
            for entity in (e1, e2):
                if not self.has_entity(entity):
                    self.entity_id_[entity] = self.eid_
                    self.id_entity_[self.eid_] = entity
                    self.entities_.add(entity)
                    self.eid_ += 1

            if e1 not in self.triples_:
                self.triples_[e1] = {}
            if r not in self.triples_[e1]:
                self.triples_[e1][r] = set()
            self.triples_[e1][r].add(e2)


    def entity_id(self, entity):
        """
        :param entity: str label
        :return eid: entity integer ID
        """
        return self.entity_id_[entity]

    def id_entity(self, eid):
        """
        :param eid: entity integer ID
        :return entity: str label
        """
        return self.id_entity_[eid]

    def csr_matrix(self):
        """
        :return A: scipy sparse CSR adjacency matrix
        """
        row, col, data = [], [], []
        for e1 in self.triples_:
            for r in self.triples_[e1]:
                for e2 in self.triples_[e1][r]:
                    row.append(self.entity_id(e1))
                    col.append(self.entity_id(e2))
                    data.append(1)

        n = self.number_of_entities()
        return csr_matrix(
                (np.array(data), (np.array(row), np.array(col))),
                shape=(n,n))

    def transition_matrix(self):
        """
        :return A: scipy CSR column-stochastic transition matrix
        """
        # Create the degree matrix
        row, col, data = [], [], []
        for e1 in self.triples_:
            for r in self.triples_[e1]:
                for e2 in self.triples_[e1][r]:
                    row.append(self.entity_id(e1))
                    col.append(self.entity_id(e1))
                    data.append(1)

        n = self.number_of_entities()
        D = csr_matrix(
                (1 / np.array(data), (np.array(row), np.array(col))),
                shape=(n,n))
        return self.csr_matrix().transpose() * D

    def reset(self):
        """Sets all values to 0"""
        self.entity_value_ = defaultdict(float)
        self.triple_value_ = defaultdict(float)

    def entity_value(self, entity):
        """
        :param entity: str
        :return value: entity float value
        """
        return self.entity_value_[entity]

    def triple_value(self, triple):
        """
        :param triple: (e1, r, e2) triple
        :return value: triple float value
        """
        return self.triple_value_[triple]

    def model_user_pref(self, query_log, power=1):
        """
        :param query_log: list of queries as dicts
        :param power: number of terms in Taylor expansion
        """
        self.reset()

        # Perform random walk on the KG
        x = query_vector(self, query_log)
        M = self.transition_matrix()
        x = random_walk_with_restart(M, x, power=power)
        # x /= np.sum(x)

        # Store entity and triple values
        for eid, val in enumerate(x):
            entity = self.id_entity(eid)
            self.entity_value_[entity] = np.log(val + 1)

        for e1 in self.triples_:
            for r in self.triples_[e1]:
                for e2 in self.triples_[e1][r]:
                    triple = (e1, r, e2)
                    eid1, eid2 = self.entity_id(e1), self.entity_id(e2)
                    self.triple_value_[triple] = np.log(x[eid1] * x[eid2] + 1)

    def query_dir(self):
        raise NotImplementedError

    def topics(self):
        raise NotImplementedError

    def topic_mids(self):
        raise NotImplementedError

    def entity_names(self):
        raise NotImplementedError

class Freebase(KnowledgeGraph):

    def __init__(self, rdf_gz='webqsp-filtered-relations-freebase-rdfs.gz',
                 entity_names='all_entities.tsv', query_dir='queries/',
                 topic_dir='by-topic/', mid_dir='by-mid/'):
        """
        :param rdf_gz: filename of Freebase dump
        :param entity_names: mapping from MIDs to labels
        :param query_dir: directory where queries are saved as json
        :param topic_dir: directory where lists of query IDs by topic are stored
        :param mid_dir: directory where lists of query IDs by MID are stored
        """
        super().__init__()
        self.name_ = 'Freebase'

        self.rdf_gz_ = os.path.join(FREEBASE_DATA_DIR, rdf_gz)
        self.entity_names_ = os.path.join(FREEBASE_DATA_DIR, entity_names)
        self.query_dir_ = os.path.join(FREEBASE_DATA_DIR, query_dir)
        self.topic_dir_ = os.path.join(FREEBASE_DATA_DIR, topic_dir)
        self.mid_dir_ = os.path.join(FREEBASE_DATA_DIR, mid_dir)

    def has_fb_prefix(self, s):
        return s.startswith('<f_')

    def is_entity(self, s):
        return s.startswith('m.') or s.startswith('g.') or \
                s.startswith('<f_m.') or s.startswith('<f_g.')

    def strip_prefix(self, s):
        return s[3:-1]

    def query_dir(self):
        return self.query_dir_

    def topic_dir(self):
        return self.topic_dir_

    def mid_dir(self):
        return self.mid_dir_

    def topics(self):
        return [fname.split('.')[0] for fname in os.listdir(self.topic_dir_)]

    def topic_mids(self):
        return [fname[:-5] for fname in os.listdir(self.mid_dir_)]

    def entity_names(self):
        entity_names = {}
        with open(self.entity_names_, 'r') as f:
            next(f)
            for line in f:
                mid, name = line.rstrip().split('\t')
                entity_names[mid] = name
        return entity_names

    def load(self, head=None, strip=True):
        with gzip.open(self.rdf_gz_, 'rt') as f:
            for line in f:
                fact = tuple(line.rstrip().split('\t')[:-1])
                e1, r = fact[:2]
                e2 = ' '.join(fact[2:])

                if strip:
                    e1 = self.strip_prefix(e1) if self.has_fb_prefix(e1) else e1
                    e2 = self.strip_prefix(e2) if self.has_fb_prefix(e2) else e2
                    r = self.strip_prefix(r) if self.has_fb_prefix(r) else r

                triple = (e1, r, e2)
                self.add_triple(triple)

                if self.number_of_triples() == head:
                    return


class YAGO(KnowledgeGraph):

    def __init__(self, rdf_gz='yagoFacts.gz', query_dir='queries/', mid_dir='by-mid/'):
        """
        :param rdf_gz: YAGO dump
        :param query_dir: directory where queries are saved as json
        :param mid_dir: directory where lists of query IDs by MID are stored
        """
        super().__init__()
        self.name_ = 'YAGO'

        self.rdf_gz_ = os.path.join(YAGO_DATA_DIR, rdf_gz)
        self.query_dir_ = os.path.join(YAGO_DATA_DIR, query_dir)
        self.mid_dir_ = os.path.join(YAGO_DATA_DIR, mid_dir)

    def is_entity(self, s):
        """Only use YAGO file with entities, no values"""
        return True

    def strip(self, s):
        return re.sub(r'([^\s\w]|)+', '', s)

    def query_dir(self):
        return self.query_dir_

    def mid_dir(self):
        return self.mid_dir_

    def topic_mids(self):
        return [fname[:-5] for fname in os.listdir(self.mid_dir_)]

    def entity_names(self):
        return { entity : entity for entity in self.entities() }

    def load(self, head=None, strip=True):
        with gzip.open(self.rdf_gz_, 'rt') as f:
            for line in f:
                fact = tuple(line.rstrip().split('\t')[:-1])
                e1, r = fact[:2]
                e2 = ' '.join(fact[2:])

                if strip:
                    e1 = self.strip(e1)
                    e2 = self.strip(e2)
                    r = self.strip(r)

                if not e1 or not e2:
                    continue

                triple = (e1, r, e2)
                self.add_triple(triple)

                if self.number_of_triples() == head:
                    return


class DBPedia(KnowledgeGraph):

    def __init__(self, rdf_gz='facts.gz', query_dir='queries/', mid_dir='by-mid/'):
        """
        :param rdf_gz: YAGO dump
        :param query_dir: directory where queries are saved as json
        :param mid_dir: directory where lists of query IDs by MID are stored
        """
        super().__init__()
        self.name_ = 'DBPedia'

        self.rdf_gz_ = os.path.join(DBPEDIA_DATA_DIR, rdf_gz)
        self.query_dir_ = os.path.join(DBPEDIA_DATA_DIR, query_dir)
        self.mid_dir_ = os.path.join(DBPEDIA_DATA_DIR, mid_dir)

    def is_entity(self, s):
        return s.startswith('<') and s.endswith('>')

    def query_dir(self):
        return self.query_dir_

    def mid_dir(self):
        return self.mid_dir_

    def topic_mids(self):
        return [fname[:-5] for fname in os.listdir(self.query_dir_)]

    def entity_names(self):
        return { entity : entity for entity in self.entities() }

    def load(self, head=None, strip=True):
        with gzip.open(self.rdf_gz_, 'rt') as f:
            for line in f:
                fact = line.rstrip('\n')[:-2].split(' ')
                e1, r = fact[:2]
                e2 = ' '.join(fact[2:])

                if not e1 or not e2:
                    continue

                triple = (e1, r, e2)
                self.add_triple(triple)

                if self.number_of_triples() == head:
                    return
