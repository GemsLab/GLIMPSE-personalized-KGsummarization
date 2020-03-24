import sys

import os
import random

import numpy as np

from collections import defaultdict

from .query import generate_query, load_question, load_questions_from_file


def reuse(query_log):
    """
    :param query_log: list of dict WebQSP-style questions
    :return reuse: float in [0, 1], percentage of repeat queries

    Assumes questions with different QIDs are unique,
    even if the questions have the same semantic meaning
    """
    unique_qids = {
        question['QuestionId'] for question in query_log
    }
    return 1 - len(unique_qids) / len(query_log)

def entity_counts(query_log):
    """
    :param query_log: list of dict WebQSP-style questions
    :return counts: {topic_entity: counts}
    """
    entities = defaultdict(int)
    for query in query_log:
        topic_entity = query['Parse']['TopicEntityName']
        entities[topic_entity] += 1
    return entities

def predicate_counts(query_log):
    """
    :param query_log: list of dict WebQSP-style questions
    :return counts: {predicate: counts}
    """
    relations = defaultdict(int)
    for query in query_log:
        for predicate in query['Parse']['InferentialChain']:
            relations[predicate] += 1
    return relations

def generate_queries_by_topic(KG, topic, n_topic_queries, n_topic_mids):
    """
    :param KG: KnowledgeGraph
    :param topic: name of querying topic ("art", "music")
    :param n_topic_queries: number of queries to generate
    :param n_topic_mids: number of unique topic entities in generated queries
    :return query_log: list of dict

    Assumes that there is a directory called
    <KG.topic_dir()> that contains files in the format of <topic>.list.
    Each of these files should list the IDs of queries in the topic,
    one query ID per line.
    Each query should be saved with the name <query ID>.json in
    a bdirectory called <KG.query_dir()>.

    Example directory structure:

    <KG.topic_dir()>/
        art.list
        music.list
        geography.list
    <KG.query_dir()>/
        q1.json
        q2.json
        q3.json
        q4.json
        q5.json

    Assuming that q1 and q3 are about art, the art.list file
    should contain the following:
        q1
        q3
    """
    # Generate the number of queries per topic MID
    p = np.random.uniform(size=n_topic_mids)
    p /= np.sum(p)
    queries_per_mid = np.int64(np.ceil(p * n_topic_queries))

    # Get all the queries that belong to the specified topic, and
    # randomly select n_topic_mids topic MIDs from the retrieved queries
    topic_file = os.path.join(KG.topic_dir(), '{}.list'.format(topic))
    queries_of_topic = load_questions_from_file(KG.query_dir(), topic_file)
    topic_mids = [
        question['Parse']['TopicEntityMid'] for qid, question in queries_of_topic.items()
    ]
    topic_mids = random.choices(topic_mids, k=n_topic_mids)

    # Obtain a selection of queries for each topic MID
    query_log = []
    for topic_mid, n_mid_queries in zip(topic_mids, queries_per_mid):
        n_mid_queries = min(n_mid_queries, n_topic_queries - len(query_log))
        query_log.extend(
                generate_queries_by_mid(KG, topic_mid, n_mid_queries)
        )

    return query_log

def generate_queries_by_mid(KG, topic_mid, n_mid_queries):
    """
    :param KG: KnowledgeGraph
    :param topic_mid: topic entity of query
    :param n_mid_queries: number of queries to generate
    :return query_log: list of queries (dict)

    Assumes that there is a directory called <KG.mid_dir()> that
    contains files in the format of <topic_mid>.list.
    Each of these files should list the IDs of queries with the
    specified topic MID, one query ID per line.
    Each query should be saved with the name <query ID>.json in
    a directory called <KG.query_dir()>.

    Example directory structure:

    <KG.mid_dir()>/
        m123e9.list
        g1048d.list
        m5ehk3.list
    <KG.query_dir()>/
        q1.json
        q2.json
        q3.json
        q4.json
        q5.json

    Assuming that q1 and q3 have topic MID m123e9,
    the file m123e9.list should contain the following:
        q1
        q3

    If the following requirements are not met, synthetic queries with the
    specified topic mid are generated.
    """
    # Obtain a selection of queries for each topic MID
    mid_file = os.path.join(KG.mid_dir(), '{}.list'.format(topic_mid))

    if os.path.isfile(mid_file):
        mid_queries = [
            question for question in load_questions_from_file(
                KG.query_dir(), mid_file).values()
        ]
        return random.choices(mid_queries, k=n_mid_queries)

    return [
        generate_query(KG, topic_mid, chain_len=random.randint(1, 3)) for _ in range(n_mid_queries)
    ]

def randomize_log(KG, query_log, random_query_prob=0.1, shuffle=False):
    """
    :param KG: KnowledgeGraph
    :param query_log: list of dict queries
    :param random_query_prob: prob. of replacing a query with a random one
    :param shuffle: randomly shuffle the returned log
    :return query_log: updated query log
    """
    n_random = np.int64(random_query_prob * len(query_log))
    indices = np.random.randint(len(query_log), size=n_random)

    query_fnames = os.listdir(KG.query_dir())

    # Add randomly selected queries at specified indices
    for index in indices:
        query_fname = random.choice(query_fnames)
        query_log[index] = load_question(os.path.join(KG.query_dir(), query_fname))

    # Randomly shuffle the log
    if shuffle:
        random.shuffle(query_log)

    return query_log

def query_log_by_topics(KG, topics, n_mids_per_topic, n_queries_in_log,
        topic_dist=None, shuffle=False, random_query_prob=0.1):
    """
    :param KG: KnowledgeGraph
    :param topics: high-level topics in the log ("art", "music")
    :param n_mids_per_topic: number of distinct topic entities per topic
    :param n_queries_in_log: number of queries in the log
    :param topic_dist: if specified, a probability distribution per topic
    :param shuffle: randomly shuffle the returned queries
    :param random_query_prob: prob. of replacing a query with a random one
    :return query_log: list of query dicts
    """
    # Number of queries per topic
    n_topics = len(topics)
    topic_dist = np.random.uniform(size=n_topics) if topic_dist is None else topic_dist
    topic_dist /= np.sum(topic_dist)
    queries_per_topic = np.int64(np.ceil(topic_dist * n_queries_in_log))

    # Get queries for each topic
    query_log = []
    for n_topic_queries, topic in zip(queries_per_topic, topics):
        n_topic_queries = min(n_topic_queries, n_queries_in_log - len(query_log))
        query_log.extend(generate_queries_by_topic(
            KG, topic, n_topic_queries, n_mids_per_topic))

    # Replace some queries with random ones
    query_log = randomize_log(KG, query_log,
        random_query_prob=random_query_prob, shuffle=shuffle)

    return query_log

def query_log_by_mids(KG, topic_mids, n_queries_in_log,
        topic_dist=None, shuffle=False, random_query_prob=0.1):
    """
    :param KG: KnowledgeGraph
    :param topic_mids: topic entities in the log
    :param n_queries_in_log: number of queries in the log
    :param topic_dist: if specified, a probability distribution per topic
    :param shuffle: randomly shuffle the returned queries
    :param random_query_prob: prob. of replacing a query with a random one
    :return query_log: list of query dicts
    """
    n_topics = len(topic_mids)
    topic_dist = np.random.uniform(size=n_topics) if topic_dist is None else topic_dist
    topic_dist /= np.sum(topic_dist)
    queries_per_topic = np.int64(np.ceil(topic_dist * n_queries_in_log))

    # Get queries for each topic mid
    query_log = []
    for n_mid_queries, topic_mid in zip(queries_per_topic, topic_mids):
        n_mid_queries = min(n_mid_queries, n_queries_in_log - len(query_log))
        query_log.extend(generate_queries_by_mid(
            KG, topic_mid, n_mid_queries))

    # Replace some queries with random ones
    query_log = randomize_log(KG, query_log,
        random_query_prob=random_query_prob, shuffle=shuffle)

    return query_log
