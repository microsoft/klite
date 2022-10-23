# Construct the query and Look up the wiktionary

```
load_wiki/
|–– download_wiktionary.sh # download the raw wiktionary json file
|–– construct_wiktionary.py # pre-process raw wiktionary json into dict wiktionary file
|–– query_wiktionary.py # use 'class_name' to query the pre-processed dict wiktionary
|–– build_dataset.py # build image-text-knowledge dataset using wiktionary and image-text pairs
|–– prompt_engineering.py # prompt for image-label dataset
|–– wiki_dict.10.json # example pre-processed dict wiktionary
```