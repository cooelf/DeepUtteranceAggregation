#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print("Useing: python train_word2vec_model.py input_text "
              "output_gensim_model output_word_vector")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    sentences = []
    for line in open(inp):
        texts = line.decode("utf-8").replace("\n","").split("\t")[1:]
        for uter in texts:
            sentences.append(uter.split())

    model = Word2Vec(sentences, size=200, window=5, min_count=0,sg=1,
                     workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(outp, binary=False)
