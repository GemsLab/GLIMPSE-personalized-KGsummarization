# GLIMPSE: Personalized Knowledge Graph Summarization

This is a reference implementation for our IEEE ICDM 2019 paper:

> Personalized Knowledge Graph Summarization: From the Cloud to Your Pocket.
  Tara Safavi, Caleb Belth, Lukas Faber, Davide Mottin, Emmanuel Muller, Danai Koutra.
  IEEE International Conference on Data Mining (ICDM), 2019

*Link*: https://gemslab.github.io/papers/safavi-2019-glimpse.pdf
  
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
- [Freebase](https://developers.google.com/freebase/), specifically the latest GZ file from the Freebase data dump. In our paper we used a parsed, cleaned version of the raw dump using the triple shrinking scripts from [FreebaseTools](https://www.isi.edu/isd/LOOM/kres/freebase-tools/). 

Note that the code to read in each knowledge graph expects ``.gz`` files, so you should gzip the raw data dumps as necessary.

In lines 13-15 of ``base.py``, change the paths to each dataset to your local data directories. 
Each subclass of ``KnowledgeGraph`` also has several keyword arguments, which you may need to change according to your directory structure and file naming conventions:

- ``rdf_gz``: The filename of the data dump in gzip format. 
- ``query_dir``: The subdirectory where generated queries are saved and retrieved in json format (see below).
- ``by_topic``: The subdirectory that stores files listing queries by topic (see below).
- ``by_mid``: The subdirectory that stores files listing queries by topic entity MID (see below).

Here's an example of how queries might be stored according to this subdirectory structure:
```
<kg_data_dir>/
  <by_topic>/
      art.list
      music.list
      geography.list
  <by_mid>/
      m934sk.list
      g104n1.list
      m10394.list
  <query_dir>/
      q1.json
      q2.json
      q3.json
      q4.json
      q5.json
```
Now, assuming that queries q1 and q3 are about "art", the ``art.list`` file should look like this:
```
q1
q3
```
Similarly, assuming that queries q1, q4, and q5 have topic entity MID ``m934sk``, the ``m934sk.list`` file should look like this:
```
q1
q4
q5
```
In essence, each of the .list files points to queries that fall under its topic/topic entity.
        
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
