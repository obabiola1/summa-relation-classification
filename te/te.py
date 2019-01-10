
import tensorflow as tf
import os
import math
import importlib

from te.model import MyModel,get_minibatch
from utils.data_processing import *
from utils.settings import CONFIG
from utils.utils import RELATIONS2DESCRIPTIONS


def make_predictions(candidates,topn=10,language="en"):
    model_key = "modeldir_"+language
    modeldir = CONFIG["MODEL"][model_key]
    model_name = CONFIG["MODEL"]["model_name"]
    dictpath = modeldir+"/"+model_name+".p"
    best_checkpoint = modeldir+"/"+model_name+".ckpt_best"
    seq_length = int(CONFIG["MODEL"]["seq_length"])
    word_embedding_dim = int(CONFIG["MODEL"]["word_embedding_dim"])
    hidden_embedding_dim = int(CONFIG["MODEL"]["hidden_embedding_dim"])
    emb_train = True if int(CONFIG["MODEL"]["emb_train"]) else False
    emb_train_topn = int(CONFIG["MODEL"]["emb_train_topn"])
    batch_size = int(CONFIG["MODEL"]["batch_size"])

    word_indices = pickle.load(open(dictpath, "rb"))
    loaded_embeddings = loadEmbedding_rand(CONFIG["MODEL"]["embedding_data_path"], word_indices)

    sentences_to_padded_index_sequences(word_indices, [candidates])

    model = MyModel(seq_length=seq_length, emb_dim=word_embedding_dim,hidden_dim=hidden_embedding_dim,
                    embeddings=loaded_embeddings,emb_train=emb_train, emb_train_topn=emb_train_topn)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    saver.restore(sess, best_checkpoint)

    total_batch = math.ceil(len(candidates) / batch_size)
    logits = np.empty(2)
    for i in range(total_batch):
        minibatch_premise_vectors, minibatch_hypothesis_vectors = get_minibatch(candidates,
        batch_size * i,batch_size * (i + 1))
        feed_dict = {model.premise_x: minibatch_premise_vectors,
                     model.hypothesis_x: minibatch_hypothesis_vectors,
                     model.keep_rate_ph: 1.0}
        logit = sess.run(model.logits, feed_dict)
        logits = np.vstack([logits, logit])
    tf.reset_default_graph()

    logits = logits[1:]
    predictions = np.argmax(logits, axis=1)
    for index, candidate in enumerate(candidates):
        candidate["score_positive"] = logits[index][0]
        candidate["score_negative"] = logits[index][1]
        candidate["prediction"] = predictions[index]

    # filter candidates and compute score
    candidates = [cand for cand in candidates if (cand["prediction"] == 0)]
    total_score = 0.0
    for candidate in candidates:
        total_score += candidate["score_positive"]
    for candidate in candidates:
        candidate["confidence_estimate"] = candidate["score_positive"]/total_score
        del candidate["score_positive"]
        del candidate["score_negative"]
        del candidate["prediction"]
        del candidate["premise_index_sequence"]
        del candidate["hypothesis_index_sequence"]

    candidates = sorted(candidates,key=lambda x:x["confidence_estimate"],reverse=True)


    candidates_final = []
    candidate_triples = set([])

    for candidate in candidates:
        triple = (candidate["subject_entity"],candidate["relation"],candidate["object_entity"])
        if not (triple in candidate_triples):
            candidates_final.append(candidate)
            candidate_triples.add(triple)


    return candidates_final[:topn]











