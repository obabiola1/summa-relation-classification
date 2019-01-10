INTERNAL API CALL SAMPLE RESPONSE FORMAT:

```
...
"sentence_token_starts": [
        0,
        21
    ]
...
```


EXTERNAL API CALL SAMPLE RESPONSE FORMAT:

```
{
    "entities": [
        {
            "entity": {
                "baseForm": "NIL0000090",
                "currlangForm": "NIL0000090",
                "id": "NIL0000090",
                "type": "organization"
            },
            "mentions": [
                {
                    "endPosition": {
                        "chunk": 0,
                        "offset": 7
                    },
                    "ner_type": "organization",
                    "souceDocument": {
                        "id": "0000000001",
                        "language": "en"
                    },
                    "startPosition": {
                        "chunk": 0,
                        "offset": 0
                    },
                    "text": "US Vice"
                }
            ]
        },
        {
            "entity": {
                "baseForm": "Donald Trump",
                "currlangForm": "Donald Trump",
                "id": "m.0cqt90",
                "type": "people"
            },
            "mentions": [
                {
                    "endPosition": {
                        "chunk": 0,
                        "offset": 197
                    },
                    "ner_type": "people",
                    "souceDocument": {
                        "id": "0000000001",
                        "language": "en"
                    },
                    "startPosition": {
                        "chunk": 0,
                        "offset": 185
                    },
                    "text": "Donald Trump"
                }
            ]
        },
        {
            "entity": {
                "baseForm": "Mohamed Abdelwahab Abdelfattah",
                "currlangForm": "Mohamed Abdelwahab Abdelfattah",
                "id": "m.03y0m68",
                "type": "people"
            },
            "mentions": [
                {
                    "endPosition": {
                        "chunk": 0,
                        "offset": 291
                    },
                    "ner_type": "people",
                    "souceDocument": {
                        "id": "0000000001",
                        "language": "en"
                    },
                    "startPosition": {
                        "chunk": 0,
                        "offset": 279
                    },
                    "text": "Abdel Fattah"
                }
            ]
        },
        {
            "entity": {
                "baseForm": "Mike Pence",
                "currlangForm": "Mike Pence",
                "id": "m.022r9r",
                "type": "people"
            },
            "mentions": [
                {
                    "endPosition": {
                        "chunk": 0,
                        "offset": 28
                    },
                    "ner_type": "people",
                    "souceDocument": {
                        "id": "0000000001",
                        "language": "en"
                    },
                    "startPosition": {
                        "chunk": 0,
                        "offset": 18
                    },
                    "text": "Mike Pence"
                }
            ]
        }
    ]
}

```