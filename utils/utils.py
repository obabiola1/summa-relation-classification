import json
from collections import defaultdict
import numpy as np


RELATIONS2DESCRIPTIONS = defaultdict(list)
with open("relation_descriptions.txt",encoding="utf-8") as input_fh:
    for line in input_fh:
        relation, description = line.strip().split("\t")
        RELATIONS2DESCRIPTIONS[relation].append(description)


class sentence:
    def __init__(self,start_offset,end_offset,text):
        self.text = text
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.mentions = []  # list of mention objects
        self.entity2mentions = defaultdict(list)

    def get_entity_count(self):
        entities = set([])
        for mention in self.mentions:
            entities.add(mention.entity_name)
        return len(entities)

    def get_entities(self):
        entities = set([])
        for mention in self.mentions:
            entities.add(mention.entity_name)
        return entities

    def __repr__(self):
        return self.text

class mention:
    def __init__(self,start_offset,end_offset,text,entity_name):
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.text = text
        self.entity_name = entity_name

    def __repr__(self):
        return self.text

class document:
    def __init__(self, doc_id, lang="en"):
        self.language = lang
        self.document_id = doc_id
        self.sentences = []


class relation:
    TYPE_SURFACE_PATTERN, TYPE_FREEBASE, TYPE_TAC = range(1, 4)
    def __init__(self, name, typ):
        self.name = name
        self.relation_type = typ


class entity_pair:
    def __init__(self, entity_one, entity_two):
        self.entity_one = entity_one
        self.entity_two = entity_two

    def get_name(self):
        return self.entity_one + "-" + self.entity_two




class MyEncoder(json.JSONEncoder):
    """
    custom json encoder
    https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)