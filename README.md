# Summa relation classification


Code which provides an API that can be used to extract relations given their descriptions, based on 
the 
approach described in [1]. Several 
components are borrowed and/or adapted from the MultiNLI (https://github.com/nyu-mll/multiNLI) code used to establish
 baselines for the MultiNLI corpus[2].

## Setup
This setup instructions assume you already setup Priberam's fork of Turbo Parser (https://github.com/Priberam/TurboParser) to process documents at the URL address `URL_INTERNAL`, an entity linker (we currently use 
Priberam's internal entity linker) to process documents at the URL address `URL_EXTERNAL`. Both URLs should then be 
specified in the `config.ini` configuration file, under 
 `[SYSTEM]`. The system assumes that the response of calls to `URL_INTERNAL` and `URL_EXTERNAL` follows a specific format. For an example of what the format looks like, see `sample.md`.

Sample relations and their descriptions, based on the question templates used in [3], are provided in the  
`relation_descriptions.txt` file.
 To extract new 
relations, edit the file by adding your own relations and their descriptions. 
 
After cloning this repository, download ` glove.840B.300d.zip` from `https://nlp.stanford.edu/projects/glove`, 
extract it and put it under the data directory. Also download our provided pretrained `model` folder from 
`https://drive.google
.com/drive/folders/15_mxp9lzE_FlzbYM-14uH6sWu4W29Tkt?usp=sharing` and put it under the `data` directory.

Then build the docker image that serves the api with the command:

```
docker build -t summaleta/relation-extraction:latest .

```
To start the API server, run it with the command:
```
docker run -it -p 7009:7009 summaleta/relation-extraction:latest

```
## Making HTTP requests to the API

Assuming the running docker container instance is accessible at `127.0.0.1`, the API endpoint is at :

`http://127.0.0.1:7000/triples/get_triples`

When making requests, a SUMMA multilingual document instance or just plain text should be provided in the body of 
your HTTP
POST request.

A JSON representation of the response is returned. A confidence estimate/score is associated with each predicted 
relation, so you can set a threshold to adjust precision/recall.
  
  
#### References
[1] Abiola Obamuyide and Andreas Vlachos."Zero-shot Relation Classification as Textual Entailment." In 
Proceedings of FEVER @ EMNLP 2018.

[2] Adina Williams, Nikita Nangia and Samuel Bowman. "A Broad-Coverage Challenge Corpus for Sentence Understanding 
through Inference." In Proceedings of NAACL 2018.

[3] Omer Levy, Minjoon Seo, Eunsol Choi and Luke Zettlemoyer. "Zero-Shot Relation Extraction via Reading Comprehension."
 In Proceedings of CoNLL 2017.