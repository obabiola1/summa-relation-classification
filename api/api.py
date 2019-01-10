#!/usr/bin/python
import configparser,json,itertools
from flask import Flask, abort, request,Response
import requests as httprequests
from collections import OrderedDict

from te.te import make_predictions
from utils.utils import RELATIONS2DESCRIPTIONS
from utils.constants import SUBJECT_PLACEHOLDER,OBJECT_PLACEHOLDER
from utils.settings import CONFIG

from utils.utils import sentence as sentence_class
from utils.utils import mention as mention_class
from utils.utils import MyEncoder

app = Flask(__name__)


@app.route('/triples/get_triples',methods=['POST'])
def getRelations(language="en",topn=10,encoding="utf-8"):
    """
    """
    document = (request.data).decode(encoding)
    offset2sentence = OrderedDict()
    offset2mention = OrderedDict()

    if not document:
        abort(400)
    try:
        internal_api_response,external_api_response = preprocess(document)
    except Exception as e:
        print("Exception while calling internal_process_document_post: %s\n" % str(e))
    sentence_starts_list = internal_api_response["sentence_starts"]
    for index in range(0,len(sentence_starts_list)):
        start_offset = sentence_starts_list[index]
        end_offset = ""
        if index < len(sentence_starts_list)-1:
            end_offset = sentence_starts_list[index+1]
        text = document[start_offset:end_offset] if end_offset else document[start_offset:]
        sentence = sentence_class(start_offset,end_offset,text)
        offset2sentence[(start_offset,end_offset)] = sentence

    offset2sentence = OrderedDict(sorted(offset2sentence.items(),key=lambda x:x[0][0]))

    entities = external_api_response["entities"]
    for entity in entities:
        for mention in entity["mentions"]:
            start_offset = mention["startPosition"]["offset"]
            end_offset = mention["endPosition"]["offset"]
            text = mention["text"]
            mention = mention_class(start_offset,end_offset,text,entity["entity"]["baseForm"])
            offset2mention[(start_offset,end_offset)] = mention

    offset2mention = OrderedDict(sorted(offset2mention.items(),key=lambda x:x[0][0]))

    for (sent_start_offset, sent_end_offset), sentence in offset2sentence.items():
        for (start_offset, end_offset), mention in offset2mention.items():
            if start_offset >= sent_start_offset:
                if sent_end_offset :
                    if end_offset < sent_end_offset:
                        sentence.mentions.append(mention)
                        sentence.entity2mentions[mention.entity_name].append(mention)
                else:
                    sentence.mentions.append(mention)
                    sentence.entity2mentions[mention.entity_name].append(mention)


    # filter offset2sentence
    sentences = []
    for offsets, sentence in offset2sentence.items():
        if sentence.get_entity_count() > 1:
            sentences.append(sentence)


    candidates = []
    predictions = []
    for sentence in sentences:
        entities = sentence.get_entities()
        for subject_entity,object_entity in itertools.permutations(entities,2):
            sentence_text = sentence.text
            for mention in sentence.entity2mentions[subject_entity]:
                sentence_text = sentence_text.replace(mention.text,SUBJECT_PLACEHOLDER)
            for obj_mention in sentence.entity2mentions[object_entity]:
                sentence_text = sentence_text.replace(obj_mention.text, OBJECT_PLACEHOLDER)
            # get rel descriptions and batcher
            for relation,descriptions in RELATIONS2DESCRIPTIONS.items():
                for description in descriptions:
                    candidate = OrderedDict()
                    candidate["subject_entity"] = subject_entity
                    candidate["relation"] = relation
                    candidate["object_entity"] = object_entity
                    candidate["sentence"] = sentence.text
                    candidate["premise"] = sentence_text
                    candidate["hypothesis"] = description
                    candidates.append(candidate)
    candidates = make_predictions(candidates,topn=topn,language=language)
    return Response(json.dumps(candidates,cls=MyEncoder), mimetype="application/json")



def preprocess(document):
    internal_api_response = None
    external_api_response = None
    URL_EXTERNAL = CONFIG["SYSTEM"]["URL_EXTERNAL"]
    URL_INTERNAL = CONFIG["SYSTEM"]["URL_INTERNAL"]
    try:
        external_payload = [{"text":document}]
        internal_api_response = httprequests.post(URL_INTERNAL, data=document).json()
        external_api_response = httprequests.post(URL_EXTERNAL, json=external_payload).json()
    except Exception as e:
        print("Exception while calling preprocessing API: %s\n" % str(e))
        raise
    return (internal_api_response,external_api_response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7009)


