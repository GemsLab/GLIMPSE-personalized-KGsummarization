import numpy as np


def query_vector(KG, query_log):
    """
    :param KG: KnowledgeGraph
    :param query_log: list of queries in dict format
    :return x: query vector (n_entities,)
    """
    x = np.zeros(KG.number_of_entities())
    for query in query_log:
        parse = query['Parse']
        topic_eid = KG.entity_id(parse['TopicEntityMid'])
        x[topic_eid] += 1
    return x

def random_walk_with_restart(M, x, c=0.15, power=1):
    """
    :param M: scipy sparse transition matrix
    :param x: np.array (n_entities,) seed initializations
    :param c: float in [0, 1], optional restart prob
    :param power: number of terms in Taylor expansion
    :return r: np.array (n_entities,) random walk vector

    Approximates the matrix inverse using the Taylor expansion:
        (I - M)^-1 = I + M + M^2 + M^3 ...
    """
    q = c * np.copy(x)
    r = np.copy(q) # result vector

    for _ in range(power):
        q = (1 - c) * M * q
        r += q
        r /= np.sum(r)
    return r
