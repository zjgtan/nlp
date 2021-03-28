import src.summarize.config as config
import numpy as np
import random
import glob
import struct
import csv
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequence


class Example:
    def __init__(self, enc_input, enc_len, dec_input, dec_len, dec_output):
        self.enc_input = enc_input
        self.enc_len = enc_len
        self.dec_input = dec_input
        self.dec_len = dec_len
        self.dec_output = dec_output

class Vocab(object):
  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          break


  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    return self._count

  def write_metadata(self, fpath):
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in range(self.size()):
        writer.writerow({"word": self._id_to_word[i]})



class CnnDmDataset:
    def __init__(self, data_path, vocab_path, max_vocab_size):
        self.data_path = data_path
        self.vocab = Vocab(vocab_path, max_vocab_size)

    def split_abstract_to_sents(self, abstract):
        cur = 0
        sents = []
        while True:
            try:
                start_p = abstract.index(SENTENCE_START, cur)
                end_p = abstract.index(SENTENCE_END, start_p + 1)
                cur = end_p + len(SENTENCE_END)
                sents.append(abstract[start_p + len(SENTENCE_START):end_p])
            except ValueError as e:  # no more sentences
                return sents

    def parse_article(self, article):
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        enc_len = len(article_words)
        enc_input = [self.vocab.word2id(w) for w in article_words]

        return enc_len, enc_input

    def parse_abstract_sentences(self, abstract_sentences):
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abstract_ids = [self.vocab.word2id(w) for w in abstract_words]
        dec_input = [self.vocab.word2id(START_DECODING)] + abstract_ids[:]
        target = abstract_ids[:]
        if len(dec_input) > config.max_dec_steps:
            dec_input = dec_input[:config.max_dec_steps]
            target = target[:config.max_dec_steps]
        else:
            target.append(self.vocab.word2id(STOP_DECODING))

        return len(dec_input), dec_input, target

    def build_example(self, article, abstract_sentences):
        enc_len, enc_input = self.parse_article(article)
        dec_len, dec_input, dec_output = self.parse_abstract_sentences(abstract_sentences)

        return {"enc_input": enc_input, "enc_len": enc_len,
                "dec_input": dec_input, "dec_len": dec_len, "dec_output": dec_output}

    def get_example(self):
        """
        解析tf.example
        :return:
        """
        filelist = glob.glob(self.data_path)
        random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    break  # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                article, abstract = example_pb2.Example.FromString(example_str)
                abstract_sentences = self.split_abstract_to_sents(abstract)
                yield self.build_example(article, abstract_sentences)

    def pad_seq(self, seq, max_len):
        padded_seq = np.zeros((config.batch_size, max_len))
        padded_seq

    def build_batch(self, example_list):
        padded_batch = {}
        padded_batch["enc_input"] = self.pad_seq([example["enc_input"] for example in example_list], config.max_enc_steps)
        padded_batch["dec_input"] = self.pad_seq([example["dec_input"] for example in example_list], config.max_dec_steps)
        padded_batch["dec_output"] = self.pad_seq([example["dec_output"] for example in example_list], config.max_dec_steps)

        return padded_batch

    def get_next(self):
        example_list = []
        for idx, example in enumerate(self.get_example()):
            if len(example_list) >= config.batch_size:
                yield self.build_batch(example_list)
            example_list.append(example)


