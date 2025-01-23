# Dataset



| Dataset  | No. Document | No. Query |
| -------- | ------------ |-----------|
| Quora    | 523K         | 10,000    |
| Lotte    | 2.4M         | 2,931     |
| HotpotQA | 5.23M        | 7,405     |
| MS MARCO | 8.84M        | 6,980     |



### Quora

This is used for duplicated question retrieval

Downloaded from https://huggingface.co/datasets/BeIR/beir

Link https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip

Put the raw dataset in /home/xxx/Dataset/vector-set-similarity-search/RawData/quora

Run `python3 preprocess_dataset/quora.py` to generate the groundtruth file



### Lotte-lifestyle

https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz

Put the raw dataset in /home/xxx/Dataset/vector-set-similarity-search/RawData/lotte-lifestyle

File needed to be transformed:

Note that the base directory is /home/xxx/Dataset/vector-set-similarity-search/RawData/lotte-lifestyle

| Original file                             | Transformed file           |
| ----------------------------------------- | -------------------------- |
| lotte/lifestyle/dev/collection.tsv        | document/collection.tsv    |
| lotte/lifestyle/dev/questions.search.tsv  | document/queries.dev.tsv   |
| lotte/lifestyle/test/questions.search.tsv | document/queries.train.tsv |

Then run `python3 preprocess_dataset/lotte_lifestyle.py` to generate the groundtruth file

### Lotte

similar as Lotte-lifestyle

Put the raw dataset in /home/xxx/Dataset/vector-set-similarity-search/RawData/lotte

https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz

File needed to be transformed manually:

Note that the base directory is /home/xxx/Dataset/vector-set-similarity-search/RawData/lotte

| Original file                          | Transformed file           |
| -------------------------------------- | -------------------------- |
| lotte/pooled/dev/collection.tsv        | document/collection.tsv    |
| lotte/pooled/dev/questions.search.tsv  | document/queries.dev.tsv   |
| lotte/pooled/test/questions.search.tsv | document/queries.train.tsv |

Then run `python3 preprocess_dataset/lotte.py` to generate the groundtruth file

### HotpotQA

downloaded from https://huggingface.co/datasets/BeIR/beir

link https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip





### MS MARCO 

https://microsoft.github.io/msmarco/Datasets.html
you should download:  collectionandqueries.tar.gz 2.9GB
use: collection.tsv, qrels.dev.small.tsv as the groundtruth and queries.dev.small.tsv as the testing query
top-k of queries.train.tsv -> queries.train.tsv



### Wikipedia

this dataset is formatted

contains two question dataset, Natural Question (NQ) and Trivia QA
NaturalQuestions: https://ai.google.com/research/NaturalQuestions/download
wiki dataset extractor: https://github.com/attardi/wikiextractor
wiki original dataset source: https://archive.org/download/enwiki-20181220

You can download the dataset of NQ and TQA in the repository of DPR
To prepare the file structure, here is the mapping from the dpr file to the file in RawData folder
data needed in Natural Question:
gold_passages_info/nq_test.json -> wiki-nq/nq_test.json
wikipedia-split/psgs_w100.tsv -> wiki-nq/collections.tsv
retriever/qas/nq-test.csv -> wiki-nq/nq-test.csv
retriever/qas/nq-train.csv -> (select top-100 of) wiki-nq/nq-train.csv

data needed in Trivia QA
wikipedia-split/psgs_w100.tsv -> wiki-tqa/collections.tsv
retriever/qas/trvia-test.csv -> wiki-tqa/tqa-test.csv

then run the script in script/preprocess_raw_data/wiki_nq.py



### MS MARCOv2

https://microsoft.github.io/msmarco/TREC-Deep-Learning-2022

\# passage: 138M

you should download: msmarco_v2_passage.tar 20.3GB, passv2_dev_queries.tsv 160.7KB, passv2_dev_top100.txt.gz 4.7MB

