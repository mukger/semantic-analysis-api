from nlp_rake import Rake


class TextProcessor:
    def __init__(self, min_chars=3, max_words=3, language_code='en'):
        self.rake = Rake(min_chars=min_chars, max_words=max_words, language_code=language_code)

    def find_key_words(self, text):
        keywords = self.rake.apply(text)
        result_dict = {word[0]: word[1] for word in keywords[:10]}
        return result_dict
