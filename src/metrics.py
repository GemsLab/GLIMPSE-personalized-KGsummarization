import numpy as np

from .query import answer_query


def precision(tp, fp, fn):
    """Precision score"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fp, fn):
    """Recall score"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(tp, fp, fn):
    """
    :param tp, fp, fn: number of true pos, false pos, false neg
    :return f1_score: harmonic mean of precision and recall
    """
    P = precision(tp, fp, fn)
    R = recall(tp, fp, fn)
    return (2 * P * R) / (P + R) if P and R else 0

def query_metrics(S, query):
    """
    :param S: Summary
    :param query: query in WebQSP format
    :return f1_score, precision, recall: metrics
    """
    kg_matches = {
        answer['AnswerArgument'] for answer in query['Parse']['Answers']
    }
    summary_matches = answer_query(S, query)

    tp = len(summary_matches.intersection(kg_matches))
    fp = len(summary_matches.difference(kg_matches))
    fn = len(kg_matches.difference(summary_matches))
    return f1_score(tp, fp, fn), precision(tp, fp, fn), recall(tp, fp, fn)

def total_query_log_metrics(S, query_log):
    """
    :param S: Summary
    :param query_log: list of queries
    :return f1_score, precision, recall: computed over entire log
    """
    tp = fp = fn = 0
    for query in query_log:
        kg_matches = answer_query(S.parent(), query)
        summary_matches = answer_query(S, query)

        tp += len(summary_matches.intersection(kg_matches))
        fp += len(summary_matches.difference(kg_matches))
        fn += len(kg_matches.difference(summary_matches))
    return f1_score(tp, fp, fn), precision(tp, fp, fn), recall(tp, fp, fn)

def average_query_log_metrics(S, query_log):
    """
    :param S: Summary
    :param query_log: list of queries
    :return f1_score, precision, recall: averaged over individual queries
    """
    f1, prec, rec = [], [], []
    for query in query_log:
        F1, P, R = query_metrics(S, query)
        f1.append(F1)
        prec.append(P)
        rec.append(R)
    return np.mean(f1), np.mean(prec), np.mean(rec)
