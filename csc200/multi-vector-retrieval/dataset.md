# Dataset



| Dataset  | No. Document | No. Query |
| -------- | ------------ |-----------|
| Quora    | 523K         | 10,000    |
| Lotte    | 2.4M         | xxx       |
| HotpotQA | 5.23M        | 7,405     |
| MS MARCO | 8.84M        | 6,980     |



### Quora

This is used for duplicated question retrieval

Downloaded from https://huggingface.co/datasets/BeIR/beir

Link https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip

Put the raw dataset in /home/xxx/Dataset/multi-vector-retrieval/RawData/quora


### Lotte

similar as Lotte-lifestyle

Put the raw dataset in /home/xxx/Dataset/multi-vector-retrieval/RawData/lotte

https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz

File needed to be transformed manually:

Note that the base directory is /home/xxx/Dataset/multi-vector-retrieval/RawData/lotte

| Original file                          | Transformed file           |
| -------------------------------------- | -------------------------- |
| lotte/pooled/dev/collection.tsv        | document/collection.tsv    |
| lotte/pooled/dev/questions.search.tsv  | document/queries.dev.tsv   |
| lotte/pooled/test/questions.search.tsv | document/queries.train.tsv |


### HotpotQA

downloaded from https://huggingface.co/datasets/BeIR/beir

link https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip





### MS MARCO 

https://microsoft.github.io/msmarco/Datasets.html
you should download:  collectionandqueries.tar.gz 2.9GB
use: collection.tsv, qrels.dev.small.tsv as the groundtruth and queries.dev.small.tsv as the testing query
top-k of queries.train.tsv -> queries.train.tsv



