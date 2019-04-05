#!/usr/bin/env python
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import logging

import os
import ipywidgets as widgets
import tensorflow as tf
from IPython import display
from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from dragnn.python import spec_builder
from dragnn.python import load_dragnn_cc_impl  # This loads the actual op definitions
from dragnn.python import render_parse_tree_graphviz
from dragnn.python import visualization
from google.protobuf import text_format
from syntaxnet import load_parser_ops  # This loads the actual op definitions
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from tensorflow.python.platform import tf_logging as logging

def load_model(base_dir, master_spec_name, checkpoint_name):
    # Read the master spec
    master_spec = spec_pb2.MasterSpec()
    with open(os.path.join(base_dir, master_spec_name), "r") as f:
        text_format.Merge(f.read(), master_spec)
    spec_builder.complete_master_spec(master_spec, None, base_dir)
    logging.set_verbosity(logging.WARN)  # Turn off TensorFlow spam.

    # Initialize a graph
    graph = tf.Graph()
    with graph.as_default():
        hyperparam_config = spec_pb2.GridPoint()
        builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
        # This is the component that will annotate test sentences.
        annotator = builder.add_annotation(enable_tracing=True)
        builder.add_saver()  # "Savers" can save and load models; here, we're only going to load.

    sess = tf.Session(graph=graph)
    with graph.as_default():
        #sess.run(tf.global_variables_initializer())
        #sess.run('save/restore_all', {'save/Const:0': os.path.join(base_dir, checkpoint_name)})
        builder.saver.restore(sess, os.path.join(base_dir, checkpoint_name))
        
    def annotate_sentence(sentence):
        with graph.as_default():
            return sess.run([annotator['annotations'], annotator['traces']],
                            feed_dict={annotator['input_batch']: [sentence]})
    return annotate_sentence

segmenter_model = load_model("/models/Russian-SynTagRus/segmenter", "spec.textproto", "checkpoint")
parser_model = load_model("/models/Russian-SynTagRus", "parser_spec.textproto", "checkpoint")

def annotate_text(text):
    sentence = sentence_pb2.Sentence(
        text=text,
        token=[sentence_pb2.Token(word=text, start=-1, end=-1)]
    )

    # preprocess
    with tf.Session(graph=tf.Graph()) as tmp_session:
        char_input = gen_parser_ops.char_token_generator([sentence.SerializeToString()])
        preprocessed = tmp_session.run(char_input)[0]
    segmented, _ = segmenter_model(preprocessed)

    annotations, traces = parser_model(segmented[0])
    assert len(annotations) == 1
    assert len(traces) == 1
    return sentence_pb2.Sentence.FromString(annotations[0])

def parse_string_from_dragnn(sentence):
    def parse_tag(tag):
        result_dict = {}

        def remove_prefix(prefix, s):
            if s.startswith(prefix):
                return s[len(prefix):]
            else:
                raise ValueError(s + " doesn't start with " + prefix)

        def remove_suffix(suffix, s):
            if s.endswith(suffix):
                return s[:-len(suffix)]
            else:
                raise ValueError(s + " doesn't end with " + suffix)

        tag = remove_prefix("attribute { ", tag)
        tag = remove_suffix(" } ", tag)

        for part in tag.split(" } attribute { "):
            part = remove_prefix('name: "', part)
            k, part = part.split('"', 1)
            part = remove_prefix(' value: "', part)
            v, part = part.split('"', 1)
            result_dict[k] = v
        
        return result_dict

    result = annotate_text(sentence)

    result_dict = {}
    words = []
    for t in result.token:
        word_dict = {}
        word_dict['word'] = t.word
        word_dict['connection_label'] = t.label
        word_dict['connection_index'] = t.head

        tag_dict = parse_tag(t.tag)
        word_dict['pos'] = tag_dict['fPOS']
        
        words.append(word_dict)
    result_dict['words'] = words
    return jsonify(result_dict)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    else:
        app.logger.exception(e)
    return jsonify(error=repr(e)), code

@app.route('/api/pos', methods=['POST'])
def pos():
    text = request.json['text']
    return parse_string_from_dragnn(text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
