import numpy as np
import gensim
import requests

SPELLING_URL = ("https://raw.githubusercontent.com/hyperreality/"
                "American-British-English-Translator/master/data/british_spellings.json")


class WordSimCalculator:
    def __init__(self, word2vec_model_file, dimensions):
        self.british_to_american_dict = requests.get(SPELLING_URL).json()

        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            word2vec_model_file,
            binary=True
        )

        self.dimensions = dimensions

    @staticmethod
    def __calculate_vector_similarity(first_vector, second_vector):
        return (np.dot(first_vector, second_vector) /
                (np.linalg.norm(first_vector) * np.linalg.norm(second_vector)))

    def americanize(self, string):
        for british_spelling, american_spelling in self.british_to_american_dict.items():
            string = string.replace(british_spelling, american_spelling)
        return string

    def average_word_vectors(self, word_vectors, word_vector_weights=None):
        aveg_vec = []
        for i in range(self.dimensions):
            aveg_sum = 0
            aveg_den = 0
            for vector in word_vectors.keys():
                if word_vectors[vector] is None:
                    continue
                vector_weight = word_vector_weights[vector] if word_vector_weights else 1
                aveg_sum += word_vectors[vector][i] * vector_weight
                aveg_den += vector_weight
            aveg_vec.append(aveg_sum / aveg_den)
        return aveg_vec

    def determine_phrase_vector(self, phrase):
        phrase_words = phrase.split(' ')
        if len(phrase_words) == 1:
            return self.model[phrase] if phrase in self.model else None
        else:
            return self.average_word_vectors({word: (self.model[word] if word in self.model else None)
                                              for word in phrase_words})

    def average_dict_vector(self, phrase_dict):
        phrase_vectors = {phrase: self.determine_phrase_vector(phrase) for phrase in phrase_dict}
        result_vector = self.average_word_vectors(phrase_vectors, phrase_dict)
        return result_vector

    def standardize_dict(self, word_dict):
        result_word_dict = {}
        for phrase in word_dict.keys():
            key = [self.americanize(word) for word in phrase.lower().replace("’s", "")
                                                                    .replace("'s", "")
                                                                    .replace("`s", "")
                                                                    .split(' ')]
            result_word_dict[' '.join(key)] = word_dict[phrase]
        return result_word_dict

    def calculate_dicts_similarity(self, first_keyword_dict, second_keyword_dict):
        standardized_fword_dict = self.standardize_dict(first_keyword_dict)
        standardized_sword_dict = self.standardize_dict(second_keyword_dict)

        averaged_first_vec = np.array(self.average_dict_vector(standardized_fword_dict))
        averaged_second_vec = np.array(self.average_dict_vector(standardized_sword_dict))

        similarity = self.__calculate_vector_similarity(averaged_first_vec, averaged_second_vec)
        return similarity

    def determine_word_simmatrix(self, first_keyword_dict, second_keyword_dict):
        similarity_dict = {}

        for fvec_word in first_keyword_dict.keys():
            similarity_dict[fvec_word] = {}
            for svec_word in second_keyword_dict.keys():
                similarity = self.__calculate_vector_similarity(self.determine_phrase_vector(fvec_word),
                                                                self.determine_phrase_vector(svec_word))
                similarity_dict[fvec_word][svec_word] = float(similarity)

        return similarity_dict
