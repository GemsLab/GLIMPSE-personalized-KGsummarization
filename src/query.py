import sys

import os
import json
import random

from collections import defaultdict


"""All questions are assumed to be JSON following the format
outlined by the paper:
"The Value of Semantic Parse Labeling for Knowledge Base Question Answering"
Yih et al. ACL 2016.

{
    'QuestionId': <qid>,
    'Parse': {
        'TopicEntityMid': <topic mid>,
        'TopicEntityName': <potential human readable name or None>
        'InferentialChain': [<relation1>, <relation2>, ...]
        'Constraints': [ {
            'SourceNodeIndex': <int>,
            'NodePredicate': <relation>,
            'Argument': <argument>,
            'EntityName': <potential human readable name or None>
         }, ...
         ],
        'Answers': [ {
                'AnswerType': 'Entity' or 'Value',
                'AnswerArgument': <argument>,
                'EntityName': <potential human readable name or None>
            }, ...
        ]
    }
}
"""

def check_question(question):
    """
    :param question: dict question in WebQSP format
    :return: whether question is formatted correctly
    """
    return 'QuestionId' in question and 'Parse' in question and \
           'TopicEntityMid' in question['Parse'] and \
           'InferentialChain' in question['Parse'] and \
           'Answers' in question['Parse']

def load_question(query_fname):
    """
    :param query_fname: query filename
    :return question: dict question in WebQSP format
    """
    with open(query_fname, 'r') as f:
        question = json.load(f)

    # Make sure the query is formatted correctly
    if not check_question(question):
        raise ValueError('Incorrectly formatted question')

    return question

def save_question(question, query_fname):
    """
    :param question: dict question in WebQSDP format
    :param query_fname: query filename
    """
    if not check_question(question):
        raise ValueError('Incorrectly formatted question')

    with open(query_fname, 'w') as f:
        json.dump(question, f)

def save_questions_by_mid(query_dir, questions):
    """
    :param query_dir: directory where queries are stored
    :param questions: dictionary of {query ID: dict}

    Will create a subdirectory called by-mid/ in the
    specified query directory if it doesn't already exist.
    """
    mid_dir = os.path.join(query_dir, 'by-mid/')
    if not os.path.isdir(mid_dir):
        os.makedirs(mid_dir)

    # Construct lists of queries for each topic entity
    for qid, question in questions.items():
        topic_mid = question['Parse']['TopicEntityMid']
        mid_file = os.path.join(mid_dir, '{}.list'.format(topic_mid))

        with open(mid_file, 'a') as f:
            f.write('{}\n'.format(qid))

def load_questions_from_dir(query_dir):
    """
    :param query_dir: directory where queries are stored
    :return questions: dict of {query ID (str) : question (dict)}

    It's assumed that each query is stored as its
    own json file within the directory, and the
    directory doesn't contain any other files.
    """
    questions = {}
    for filename in os.listdir(query_dir):
        question = load_question(os.path.join(query_dir, filename))
        qid = question['QuestionId']
        questions[qid] = question
    return questions

def load_questions_from_file(query_dir, fname):
    """
    :param query_dir: directory where queries to load are stored
    :param fname: file listing query IDs
    :output questions: dict of {query ID (str) : question (dict)}

    It's assumed that each query is stored as its
    own json file within the directory, and the
    directory doesn't contain any other files.
    Only loads the questions stored in the fname file.
    """
    qids = load_qids(fname)
    return {qid: load_question(
        os.path.join(query_dir, '{}.json'.format(qid))) for qid in qids}

def load_qids(fname):
    """
    :param fname: file listing query IDs
    :return qid_list: list of query IDs in file
    """
    qids = []
    with open(fname, 'r') as f:
        for line in f:
            qids.append(line.rstrip())
    return qids

def is_synthetic_query(query):
    """
    :param query: query in WebQSP dict format
    :return: whether query is synthetic
    """
    return query['QuestionId'].startswith('Synth')

def is_webqsp_query(query):
    """
    :param query: query in WebQSP dict format
    :return: whether query is from WebQSP dataset
    """
    return query['QuestionId'].startswith('WebQ')

def get_name(mid, entity_names):
    """
    :param mid: entity MID (str)
    :param entity_names: dict of {MID: label}
    :return name: str or None
    """
    return entity_names[mid] if mid in entity_names else mid

def answer_query(KG, query):
    """
    :param KG: object that can be accessed by {subject: {predicate: {object}}}
    :query: query in WebQSP dict format
    :return result: set of entity answers
    """
    parse = query['Parse']
    topic_mid = parse['TopicEntityMid']
    inferential_chain = parse['InferentialChain']

    # Create a mapping of constraints to parts of the inf. chain
    constraints = defaultdict(list)
    for constraint in parse['Constraints']:
        index = constraint['SourceNodeIndex']
        constraints[index].append(constraint)

    result = {topic_mid}
    for index, predicate in enumerate(inferential_chain):
        # Add candidate answer entities
        candidates = set()
        for entity in result:
            if entity in KG and predicate in KG[entity]:
                candidates.update(KG[entity][predicate])

        # Remove candidates that don't fit the constraints
        remove = set()
        for constraint in constraints[index]:
            argument, predicate = constraint['Argument'], constraint['NodePredicate']
            for entity in candidates:
                if entity not in KG or \
                    predicate not in KG[entity] or \
                    argument not in KG[entity][predicate]:
                    remove.add(entity)

        candidates = candidates.difference(remove)
        result = candidates

    return result.difference({topic_mid}) # topic entity cannot be part of answer

def generate_query(KG, topic_mid, chain_len=2, qid=0,
        entity_names={}, constraint_index=None, exclude_preds=[]):
    """
    :param KG: object that can be accessed by {subject: {predicate: {object}}}
    :param topic_mid: str ID of the query's topic entity
    :param qid: unique identifier for this question
    :param chain_len: max number of relationships in the inferential chain
    :param entity_names: mapping of {entity ID: label}
    :param contrain_index: optional index at [0, chain_len - 1] to place a constraint
    :param exclude_preds: predicates to exclude from generated query
    :return query: query following WebQSP query structure, without metadata/comments
    """
    inferential_chain = []
    constraints = []

    # Create the core inferential chain (directed path) first
    entity = topic_mid
    for _ in range(chain_len):
        predicates = [
                pred for pred in KG[entity] if pred not in exclude_preds
        ] if entity in KG else []
        if not predicates:
            break

        predicate = random.choice(predicates)
        inferential_chain.append(predicate)

        entities = KG[entity][predicate]
        entity = random.choice(list(entities))

    # Add constraints and get the answers
    result = {topic_mid}
    for index, predicate in enumerate(inferential_chain):
        # Add candidate answer entities
        candidates = set()
        for entity in result:
            if entity in KG and predicate in KG[entity]:
                candidates.update(KG[entity][predicate])

        if candidates and constraint_index == index:
            entity = random.choice(list(candidates))
            predicates = [
                pred for pred in KG[entity] if pred not in inferential_chain \
                    and pred not in exclude_preds
            ] if entity in KG else []

            if predicates:
                predicate = random.choice(predicates)
                if KG[entity][predicate]:
                    argument = random.choice(list(KG[entity][predicate]))
                    constraints.append({
                        'SourceNodeIndex': index,
                        'NodePredicate': predicate,
                        'Argument': argument,
                        'EntityName': get_name(entity, entity_names)
                    })

                    remove = set()
                    for entity in candidates:
                        if entity not in KG or \
                            predicate not in KG[entity] or \
                            argument not in KG[entity][predicate]:
                            remove.add(entity)
                    candidates = candidates.difference(remove)

        result = candidates

    return {
        'QuestionId': str(qid),
        'Parse': {
            'TopicEntityMid': topic_mid,
            'TopicEntityName': get_name(topic_mid, entity_names),
            'InferentialChain': inferential_chain,
            'Constraints': constraints,
            'Answers': [ {
                    'AnswerType': 'Entity' if KG.is_entity(answer) else 'Value',
                    'AnswerArgument': answer,
                    'EntityName': get_name(answer, entity_names)
                } for answer in result.difference({topic_mid})
            ]
        }
    }
