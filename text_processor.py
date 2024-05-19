import nltk
from nlp_rake import Rake
from nltk.corpus import stopwords
nltk.download('stopwords')


class TextProcessor:
    def __init__(self, min_chars=3, max_words=3, language_code='en'):
        self.rake = Rake(min_chars=min_chars, max_words=max_words, language_code=language_code)
        self.stop_words = set(stopwords.words('english'))

    def find_materials_keywords(self, text):
        keywords = self.rake.apply(text)
        result_dict = {word[0]: word[1] for word in keywords[:10]}
        return result_dict

    def find_name_keywords(self, name):
        words = name.split()
        words = [word for word in words]
        keywords = [word if self.__is_abbreviation(word) else word.lower()
                    for word in words if word not in self.stop_words]
        return ' '.join(keywords)

    @staticmethod
    def __is_abbreviation(word):
        return word.isupper() and len(word) > 1

