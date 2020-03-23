#!/usr/bin/python3

# python3
# coding: utf-8


import argparse
import os
import numpy as np
import pickle
from smart_open import open
from elmo_helpers import load_elmo_embeddings, get_elmo_vectors, tf, tokenize

def make_elmo_vectors_ruwordnet(data_path, model_directory, batch_size=25):
    model_name = os.path.basename(model_directory)
    data_name  = os.path.basename(data_path).split('.')[0]
    data_dir   = os.path.dirname(data_path)

    raw_sentences = []
    with open(data_path, 'r') as f:
        for line in f:
            res = line.strip()
            raw_sentences.append(res)
    sentences = [tokenize(s) for s in raw_sentences]
    print('=====')
    print('%d sentences total' % len(sentences))
    print('=====')

    batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(model_directory)

    cropped_vectors = list()
    averaged_vectors = list()
    # Actually producing ELMo embeddings for our data:
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
        # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    for batch in [sentences[i * batch_size:(i + 1) * batch_size]
                  for i in range((len(sentences) + batch_size - 1) // batch_size)]:
        elmo_vectors_batch = get_elmo_vectors(
            sess, batch, batcher, sentence_character_ids, elmo_sentence_input)

        # print('ELMo embeddings for your input are ready')
        # print('Tensor shape:', elmo_vectors.shape)

        # Due to batch processing, the above code produces for each sentence
        # the same number of token vectors, equal to the length of the longest sentence
        # (the 2nd dimension of the elmo_vector tensor).
        # If a sentence is shorter, the vectors for non-existent words are filled with zeroes.
        # Let's make a version without these redundant vectors:
        cropped_vectors_batch = []
        for vect, sent in zip(elmo_vectors_batch, sentences):
            cropped_vector = vect[:len(sent), :]
            cropped_vectors_batch.append(cropped_vector)
            averaged_vectors.append(np.mean(cropped_vector, axis=0))

        cropped_vectors.extend(cropped_vectors_batch)

    averaged_vectors_np = np.ndarray((len(averaged_vectors),
                                      averaged_vectors[0].shape[0]),
                                     averaged_vectors[0].dtype)
    for i,avg_vector in enumerate(averaged_vectors):
        averaged_vectors_np[i] = averaged_vectors[i]

    out_filename_pckl = os.path.join(data_dir,
                                     '_'.join([data_name,'elmo_vectors', model_name])+'.pkl')
    out_filename_npy = os.path.join(data_dir,
                                     '_'.join([data_name,'elmo_avg_vectors', model_name])+'.npy')

    with open(out_filename_pckl, 'wb') as f:
        pickle.dump(cropped_vectors, f)

    with open(out_filename_npy, 'wb') as f:
        np.save(f, averaged_vectors_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to input text, one sentence per line', required=True)
    arg('--elmo', '-e', help='Path to ELMo model', required=True)
    arg('--batch_size', help='', type=int, default=25)

    args = parser.parse_args()
    data_path = args.input

    make_elmo_vectors_ruwordnet(args.input, args.elmo, args.batch_size)
