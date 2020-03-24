# glimpse-summary

This is a reference implementation for our IEEE ICDM 2019 paper:

> Personalized Knowledge Graph Summarization: From the Cloud to Your Pocket
  T. Safavi, C. Belth, L. Faber, D. Mottin, E. Muller, D. Koutra
  IEEE International Conference on Data Mining (ICDM), 2019
  
If you use it, please cite the following: 
```
@inproceedings{safavi2019personalized,
  title={Personalized Knowledge Graph Summarization: From the Cloud to Your Pocket},
  author={Safavi, Tara and Belth, Caleb and Faber, Lukas and Mottin, Davide and M{\"u}ller, Emmanuel and Koutra, Danai},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  pages={528--537},
  year={2019},
  organization={IEEE}
}
```

# Requirements

- Python 3.4 or above
- numpy
- scipy
- pandas

## Data

In our experiments we used the following datasets:

- [DBPedia 3.5.1](https://wiki.dbpedia.org/services-resources/datasets/data-set-35/data-set-351#h115-3), specifically the "Ontology Infobox Properties"  file, which is called ``mappingbased_properties_en.nt``.
- [YAGO 3](https://datahub.io/collections/yago), specifically the ``yagoFacts.tsv``, ``yagoLiteralFacts.tsv``, and ``yagoDateFacts.tsv`` files.
- [Freebase](https://developers.google.com/freebase/), specifically the latest GZ file from the Freebase data dump. 

Note that the code to read in each knowledge graph expects ``.gz`` files, so you should gzip the raw data dumps as necessary.

In lines 13-15 of ``base.py``, change the paths to each dataset to your local data directories. 
Each subclass of ``KnowledgeGraph`` also has several keyword arguments for names of data dump files and data directories where queries are saved. Change these as necessary.

## Command-line arguments

```
usage: main.py [-h] [--kg {YAGO,Freebase,DBPedia}] [--n-queries N_QUERIES]
               [--n-topic-mids N_TOPIC_MIDS] [--n-topics N_TOPICS]
               [--n-mids-per-topic N_MIDS_PER_TOPIC] [--n_users N_USERS]
               [--test-size TEST_SIZE] [--percent-triples PERCENT_TRIPLES]
               [--random-query-prob RANDOM_QUERY_PROB] [--shuffle]
               [--method {glimpse,glimpse-2} [{glimpse,glimpse-2} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --kg {YAGO,Freebase,DBPedia}
                        KG to summarize
  --n-queries N_QUERIES
                        Number of queries to simulate per user. Default is
                        200.
  --n-topic-mids N_TOPIC_MIDS
                        Number of topic mids of interest per user. Default is
                        50.
  --n-topics N_TOPICS   Number of topics to simulate per user log. For
                        Freebase only. Default is 3.
  --n-mids-per-topic N_MIDS_PER_TOPIC
                        Number of unique MIDs per topic. For Freebase only.
                        Default is 20.
  --n_users N_USERS     Number of users to simulate. Default is 5.
  --test-size TEST_SIZE
                        Percentage of queries per user to hold out for
                        testing, in [0, 1]. Default is 0.5.
  --percent-triples PERCENT_TRIPLES
                        Ratio of number of triples of KG to use as K (summary
                        constraint). Default is 0.001.
  --random-query-prob RANDOM_QUERY_PROB
                        Probability of users asking random queries rather than
                        topic-specific ones. Default is 0.1.
  --shuffle             Set this flag to true to shuffle all generated logs.
                        Default False.
  --method {glimpse,glimpse-2} [{glimpse,glimpse-2} ...]
                        Summarization methods to call. Default is [glimpse].
```
