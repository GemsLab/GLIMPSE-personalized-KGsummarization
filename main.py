import argparse
import random
import logging

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)

from time import time
from sklearn.model_selection import train_test_split

from src.base import YAGO, DBPedia, Freebase
from src.user import query_log_by_mids, query_log_by_topics
from src.glimpse import SummaryMethod, GLIMPSE
from src.metrics import total_query_log_metrics, average_query_log_metrics


# Available choices for user input arguments in main
# TODO: Change these to point to your local data directories
KG_MAPPING = {
    'YAGO': YAGO(query_dir='queries/final/', mid_dir='queries/by-mid/'),
    'Freebase': Freebase(query_dir='queries/final/'),
    'DBPedia': DBPedia()
}

METHODS = {
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
    'glimpse-2': SummaryMethod(GLIMPSE, 'GLIMPSE-2', power=2),
}

def answer_queries_in_log(KG, K, query_log, summary_methods, test_size=0.5):
    """
    :param KG: KnowledgeGraph
    :param K: summary constraint
    :param query_log: list of dict
    :param summary_methods: summarization methods to use
    :param test_size: percent of queries to hold out for testing
    """
    # Split the query log for training/testing
    train_log, test_log = train_test_split(query_log, test_size=test_size)
    logging.info('\tSplit query log into {}/{} split'.format(
        int((1 - test_size) * 100), int(test_size * 100)))

    for summary_method in summary_methods:
        logging.info('\t---Summarizing with {}---'.format(summary_method.name()))

        # Run the summarization function
        t0 = time()
        S = summary_method(KG, K, train_log) # call the object as a function
        runtime = time() - t0
        logging.info('\t  Time: {:.2f} seconds'.format(runtime))

        # Evaluate question answering on the testing queries
        total_F1, total_precision, total_recall = total_query_log_metrics(S, test_log)
        logging.info('\t  Total F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            total_F1, total_precision, total_recall))

        avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, test_log)
        logging.info('\t  Average F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            avg_F1, avg_precision, avg_recall))

def float_in_zero_one(value):
    """Check if a float value is in [0, 1]"""
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be a float between 0 and 1')
    return value

def positive_int(value):
    """Check if an integer value is positive"""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Value must be positive')
    return value

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kg', choices=list(KG_MAPPING.keys()), default='YAGO',
            help='KG to summarize')
    parser.add_argument('--n-queries', type=positive_int, default=200,
            help='Number of queries to simulate per user. Default is 200.')
    parser.add_argument('--n-topic-mids', type=positive_int, default=50,
            help='Number of topic mids of interest per user. Default is 50.')
    parser.add_argument('--n-topics', type=positive_int, default=3,
            help='Number of topics to simulate per user log. '
                 'For Freebase only. Default is 3.')
    parser.add_argument('--n-mids-per-topic', type=positive_int, default=20,
            help='Number of unique MIDs per topic. For Freebase only. Default is 20.')
    parser.add_argument('--n_users', type=positive_int, default=5,
            help='Number of users to simulate. Default is 5.')
    parser.add_argument('--test-size', type=float_in_zero_one, default=0.5,
            help='Percentage of queries per user to hold out for testing, '
                 'in [0, 1]. Default is 0.5.')
    parser.add_argument('--percent-triples', type=float_in_zero_one, default=0.001,
            help='Ratio of number of triples of KG to use as K '
                 '(summary constraint). Default is 0.001.')
    parser.add_argument('--random-query-prob', type=float_in_zero_one, default=0.1,
            help='Probability of users asking random queries rather '
                 'than topic-specific ones. Default is 0.1.')
    parser.add_argument('--shuffle', action='store_true',
            help='Set this flag to true to shuffle all generated logs. Default False.')
    parser.add_argument('--method', nargs='+', default=['glimpse'],
            choices=list(METHODS.keys()),
            help='Summarization methods to call. Default is [glimpse].')

    return parser.parse_args()

def main():
    args = parse_args()

    KG = KG_MAPPING[args.kg]
    summary_methods = [METHODS[name] for name in args.method]

    # Load the KG into memory
    logging.info('Loading {}'.format(KG.name()))
    KG.load()
    logging.info('Loaded {}'.format(KG.name()))

    # Number of triples for summary
    K = int(args.percent_triples * KG.number_of_triples())
    logging.info('K = {}'.format(K))

    # Simulate users with specified parameters
    for user in range(args.n_users):
        logging.info('---Simulating user {}---'.format(user))

        if args.kg == 'Freebase':
            topics = random.sample(KG.topics(), k=args.n_topics)

            query_log = query_log_by_topics(
                    KG, topics, args.n_mids_per_topic, args.n_queries,
                    shuffle=args.shuffle, random_query_prob=args.random_query_prob)
        else:
            topic_mids = random.sample(KG.topic_mids(), k=args.n_topic_mids)

            query_log = query_log_by_mids(
                    KG, topic_mids, args.n_queries,
                    shuffle=args.shuffle,
                    random_query_prob=args.random_query_prob)
        logging.info('---Generated a log of {} queries----'.format(len(query_log)))

        answer_queries_in_log(KG, K, query_log, summary_methods, test_size=args.test_size)

    logging.info('Shutting down...')

if __name__ == '__main__':
    main()


