#!/usr/bin/python
import os
import configparser


CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
CONFIG.read("config.ini")

RESOURCE_METHODS = ['GET']

ITEM_METHODS = ['GET']



CACHE_CONTROL = 'no-cache'
CACHE_EXPIRES = 0

triples = {
    'item_title': 'd_relation',
    'schema': {
        'subjecty': {
            'type': 'string',
            'minlength': 1,
            'maxlength': 200,
        },
        'object': {
            'type': 'string',
            'minlength': 1,
            'maxlength': 200,
        },
        'relation': {
            'type': 'string',
            'minlength':1,
            'maxlength':200,
        },
    }
}

DOMAIN = {'triples': triples,}
