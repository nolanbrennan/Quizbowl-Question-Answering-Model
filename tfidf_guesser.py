from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
import os

from typing import Union, Dict
from collections.abc import Iterable

import math
import logging
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
QN_PATH = 'questions.pickle'
ANS_PATH = 'answers.pickle'

import os

from nltk.tokenize import sent_tokenize
from guesser import print_guess, Guesser
from sklearn.feature_extraction.text import TfidfVectorizer


kTFIDF_TEST_QUESTIONS = {"This capital of England": ['Maine', 'Boston'],
                        "The author of Pride and Prejudice": ['Jane_Austen', 'Jane_Austen'],
                        "The composer of the Magic Flute": ['Wolfgang_Amadeus_Mozart', 'Wolfgang_Amadeus_Mozart'],
                        "The economic law that says 'good money drives out bad'": ["Gresham's_law", "Gresham's_law"],
                        "located outside Boston, the oldest University in the United States": ['College_of_William_&_Mary', 'Rhode_Island']}

class BPEVectorizer(TfidfVectorizer):
    def __init__(
        self,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            analyzer=analyzer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )
        self._vocab = Vocab()
        self._merges = []
        self._end_id = self._vocab.add("<END>")

    def initial_tokenize(self, sent: str):
        token_ids = []
        for b in bytearray(sent, "utf-8"):
            token_ids.append(self._vocab.add(str(b)))
        token_ids.append(self._end_id)
        return token_ids

    @staticmethod
    def merge_tokens(tokens, merge_left, merge_right, merge_id):
        i, merged = 0, []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == merge_left and tokens[i + 1] == merge_right:
                merged.append(merge_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def get_stats(self, corpus):
        pairs = defaultdict(int)
        for tokens in corpus:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def merge_vocab(self, pair, corpus):
        new_corpus = []
        left, right = pair
        new_id = self._vocab.add(f"{self._vocab.lookup_word(left)}_{self._vocab.lookup_word(right)}")
        self._merges.append((left, right, new_id))
        for tokens in corpus:
            merged = self.merge_tokens(tokens, left, right, new_id)
            new_corpus.append(merged)
        return new_corpus

    def train_bpe(self, texts, num_merges=100, min_frequency=2):
        corpus = []
        for sent in texts:
            token_ids = self.initial_tokenize(sent)
            corpus.append(token_ids)

        for _ in range(num_merges):
            stats = self.get_stats(corpus)
            if not stats:
                break
            (best_pair, freq) = max(stats.items(), key=lambda x: x[1])
            if freq < min_frequency:
                break
            corpus = self.merge_vocab(best_pair, corpus)

        self._vocab.finalize()

    def tokenize(self, sent: str):
        assert self._vocab.final
        token_ids = self.initial_tokenize(sent)
        for left_id, right_id, new_id in self._merges:
            token_ids = self.merge_tokens(token_ids, left_id, right_id, new_id)
        if token_ids[-1] != self._end_id:
            token_ids.append(self._end_id)
        return token_ids

    def build_analyzer(self):
        def analyzer(doc):
            tokens = self.tokenize(doc)
            return [str(tok) for tok in tokens]
        return analyzer

class Vocab:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.final = False
        self.next_id = 0

    def add(self, word: str) -> int:
        if word in self.word2id:
            return self.word2id[word]
        idx = self.next_id
        self.word2id[word] = idx
        self.id2word[idx] = word
        self.next_id += 1
        return idx

    def lookup_word(self, idx: int) -> str:
        return self.id2word[idx]

    def lookup_id(self, word: str) -> int:
        return self.word2id[word]

    def finalize(self):
        self.final = True


class DummyVectorizer:
    """
    A dumb vectorizer that only creates a random matrix instead of something real.
    """
    def __init__(self, width:int=50):
        self.width = width
        self.vocabulary_ = {}
    
    def transform(self, questions: Iterable):
        import numpy as np
        return np.random.rand(len(questions), self.width)

class TfidfGuesser(Guesser):


    def __init__(self, filename:str, min_df:int=1, max_df:float=1):
        """
        Initializes data structures that will be useful later.

        Args:
           filename: base of filename we store vectorizer and documents to
           min_df: we use the sklearn vectorizer parameters, this for min doc freq
           max_df: we use the sklearn vectorizer parameters, this for max doc freq
        """

        # You'll need add the vectorizer here and replace this fake vectorizer
        self.tfidf_vectorizer = BPEVectorizer(min_df=min_df, max_df=max_df, stop_words=None, token_pattern=r"(?u)\b\w+\b")
        self.tfidf = None 
        self.questions = None
        self.answers = None
        self.filename = filename


    def train(self, training_data, answer_field='page', split_by_sentence=True,
                    min_length=-1, max_length=-1, remove_missing_pages=True):
        """
        The base class (Guesser) populates the questions member, so
        all that's left for this function to do is to create new members
        that have a vectorizer (mapping documents to tf-idf vectors) and
        the matrix representation of the documents (tfidf) consistent
        with that vectorizer.
        """
        
        Guesser.train(self, training_data, answer_field, split_by_sentence, min_length,
                      max_length, remove_missing_pages)


        self.tfidf_vectorizer.train_bpe(self.questions, num_merges=200, min_frequency=2)
        self.tfidf = self.tfidf_vectorizer.fit_transform(self.questions)

        logging.info("Creating tf-idf dataframe with %i" % len(self.questions))
        
    def save(self):
        """
        Save the parameters to disk
        """
        Guesser.save_questions_and_answers(self)
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open("%s.tfidf.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf, f)
        

    def __call__(self, question, max_n_guesses):
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        question -- Raw text of the question
        max_n_guesses -- How many top guesses to return
        """
        bpe_question = " ".join(str(tok) for tok in self.tfidf_vectorizer.tokenize(question))
        question_tfidf = self.tfidf_vectorizer.transform([bpe_question])
        cosine_similarities = cosine_similarity(question_tfidf, self.tfidf)
        cos = cosine_similarities[0]
        indices = cos.argsort()[::-1]

        guesses = []
        for i in range(min(max_n_guesses, len(indices))):
            idx = indices[i]
            guess = {
                "question": self.questions[idx],
                "guess": self.answers[idx],
                "confidence": cos[idx]
            }
            guesses.append(guess)
        return guesses

    def batch_guess(self, questions:Iterable[str], max_n_guesses:int, block_size:int=1024) -> Iterable[Dict[str, Union[str, float]]]:
      """
      The batch_guess function allows you to find the search
      results for multiple questions at once.  This is more efficient
      than running the retriever for each question, finding the
      largest elements, and returning them individually.  
  
      To understand why, remember that the similarity operation for an
      individual query and the corpus is a dot product, but if we do
      this as a big matrix, we can fit all of the documents at once
      and then compute the matrix as a parallelizable matrix
      multiplication.
  
      The most complicated part is sorting the resulting similarities,
      which is a good use of the argpartition function from numpy.
  
      Args:
         questions: the questions we'll produce answers for
         max_n_guesses: number of guesses to return
         block_size: split large lists of questions into arrays of this many rows
      Returns:
      """
      
      all_guesses = []
  
      logging.info("Querying matrix of size %i with block size %i" %
                   (len(questions), block_size))
  
      for start in tqdm(range(0, len(questions), block_size)):
           stop = min(start + block_size, len(questions))
           block = questions[start:stop]
           logging.info("Block %i to %i (%i elements)" % (start, stop, len(block)))

           bpe_block = [" ".join(str(tok) for tok in self.tfidf_vectorizer.tokenize(q)) for q in block]
           block_tfidf = self.tfidf_vectorizer.transform(bpe_block)

           cosine_similarities = cosine_similarity(block_tfidf, self.tfidf)
          
          # Process each question in the block
           for question_idx in range(len(block)):
              cos = cosine_similarities[question_idx]
              
              # Get the indices of the top answers
              # Using argpartition is more efficient than full sort for getting top-k
              if max_n_guesses < len(cos):
                  # Get indices of top max_n_guesses elements
                  top_indices = np.argpartition(cos, -max_n_guesses)[-max_n_guesses:]
                  # Sort these top indices by their scores
                  top_indices = top_indices[np.argsort(cos[top_indices])[::-1]]
              else:
                  # If requesting more guesses than documents, just sort all
                  top_indices = np.argsort(cos)[::-1][:max_n_guesses]
              
              guesses = []
              for idx in top_indices:
                  guesses.append({
                      "guess": self.answers[idx], 
                      "confidence": cos[idx], 
                      "question": self.questions[idx]
                  })
              all_guesses.append(guesses)
  
      assert len(all_guesses) == len(questions), "Guesses (%i) != questions (%i)" % (len(all_guesses), len(questions))
      return all_guesses
    
    def get_stats(self, corpus):
        pairs = defaultdict(int)
        for tokens in corpus:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def merge_vocab(self, pair, corpus):
        new_corpus = []
        left, right = pair
        new_id = self._vocab.add(f"{self._vocab.lookup_word(left)}_{self._vocab.lookup_word(right)}")
        self._merges.append((left, right, new_id))
        for tokens in corpus:
            merged = self.merge_tokens(tokens, left, right, new_id)
            new_corpus.append(merged)
        return new_corpus

    def train_bpe(self, texts, num_merges=100, min_frequency=2):
        corpus = []
        for sent in texts:
            token_ids = self.initial_tokenize(sent)
            corpus.append(token_ids)

        for i in range(num_merges):
            stats = self.get_stats(corpus)
            if not stats:
                break
            (best_pair, freq) = max(stats.items(), key=lambda x: x[1])
            if freq < min_frequency:
                break
            corpus = self.merge_vocab(best_pair, corpus)

        self._vocab.finalize()

    def load(self):
        """
        Load the tf-idf guesser from a file
        """
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open("%s.tfidf.pkl" % path, 'rb') as f:
            self.tfidf = pickle.load(f)

        self.load_questions_and_answers()

if __name__ == "__main__":
    # Load a tf-idf guesser and run it on some questions
    from params import *
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    add_general_params(parser)
    add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    
    guesser = load_guesser(flags, load=True)

    questions = list(kTFIDF_TEST_QUESTIONS.keys())
    guesses = guesser.batch_guess(questions, 3, 2)

    for qq, gg in zip(questions, guesses):
        print("----------------------")
        print(qq, gg)
