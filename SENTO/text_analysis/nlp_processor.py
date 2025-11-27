import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


class NLPProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Using basic processing.")
            self.nlp = None

        self.lemmatizer = WordNetLemmatizer()

        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])

        self.emotional_entities = ['PERSON', 'ORG', 'GPE', 'EVENT']

    def preprocess_text(self, text):
        """Comprehensive text preprocessing"""
        if not text:
            return {
                'original_text': '',
                'tokens': [],
                'lemmas': [],
                'pos_tags': [],
                'entities': [],
                'sentences': [],
                'processed_text': ''
            }

        # Basic cleaning
        text = self._clean_text(text)

        if self.nlp is None:
            # Fallback processing without spaCy
            return self._basic_preprocessing(text)

        # Tokenization with spaCy
        try:
            doc = self.nlp(text)

            # Extract features
            tokens = [token for token in doc]
            lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
            pos_tags = [(token.text, token.pos_) for token in doc]
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            return {
                'original_text': text,
                'tokens': [token.text for token in tokens],
                'lemmas': lemmas,
                'pos_tags': pos_tags,
                'entities': entities,
                'sentences': [sent.text for sent in doc.sents],
                'processed_text': ' '.join(lemmas)
            }
        except Exception as e:
            print(f"spaCy processing error: {e}")
            return self._basic_preprocessing(text)

    def _basic_preprocessing(self, text):
        """Basic text processing without spaCy"""
        # Simple tokenization
        tokens = word_tokenize(text) if text else []
        sentences = sent_tokenize(text) if text else []

        # Basic lemmatization and cleaning
        lemmas = []
        for token in tokens:
            if token.lower() not in self.stop_words and token not in string.punctuation:
                try:
                    lemma = self.lemmatizer.lemmatize(token.lower())
                    lemmas.append(lemma)
                except:
                    lemmas.append(token.lower())

        return {
            'original_text': text,
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': [],  # Not available without spaCy
            'entities': [],  # Not available without spaCy
            'sentences': sentences,
            'processed_text': ' '.join(lemmas)
        }

    def _clean_text(self, text):
        """Clean text while preserving emotional content"""
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Preserve emotional punctuation
        text = re.sub(r'[^\w\s!?.,]', '', text)
        return text.strip()

    def extract_emotional_entities(self, text):
        """Extract entities that might have emotional significance"""
        if not text or self.nlp is None:
            return []

        try:
            doc = self.nlp(text)
            emotional_entities = []

            for ent in doc.ents:
                if ent.label_ in self.emotional_entities:
                    emotional_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })

            return emotional_entities
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def analyze_sentence_structure(self, text):
        """Analyze sentence structure for emotional cues"""
        if not text:
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'question_sentences': 0,
                'exclamatory_sentences': 0,
                'complex_sentences': 0
            }

        try:
            if self.nlp is not None:
                doc = self.nlp(text)
                sentences = list(doc.sents)
            else:
                sentences = [type('obj', (object,), {'text': sent}) for sent in sent_tokenize(text)]

            analysis = {
                'sentence_count': len(sentences),
                'avg_sentence_length': 0,
                'question_sentences': 0,
                'exclamatory_sentences': 0,
                'complex_sentences': 0
            }

            if sentences:
                total_words = 0
                for sent in sentences:
                    sent_text = sent.text if hasattr(sent, 'text') else str(sent)
                    words = sent_text.split()
                    total_words += len(words)

                    if sent_text.strip().endswith('?'):
                        analysis['question_sentences'] += 1
                    if sent_text.strip().endswith('!'):
                        analysis['exclamatory_sentences'] += 1
                    if len(words) > 15:  # Simple complexity measure
                        analysis['complex_sentences'] += 1

                analysis['avg_sentence_length'] = total_words / len(sentences)

            return analysis
        except Exception as e:
            print(f"Sentence structure analysis error: {e}")
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'question_sentences': 0,
                'exclamatory_sentences': 0,
                'complex_sentences': 0
            }

    def detect_emotional_patterns(self, text):
        """Detect specific emotional patterns in text"""
        if not text:
            return {
                'self_reference': 0,
                'negative_cognition': 0,
                'positive_outlook': 0,
                'uncertainty': 0
            }

        patterns = {
            'self_reference': self._detect_self_reference(text),
            'negative_cognition': self._detect_negative_cognition(text),
            'positive_outlook': self._detect_positive_outlook(text),
            'uncertainty': self._detect_uncertainty(text)
        }

        return patterns

    def _detect_self_reference(self, text):
        """Detect self-referential language"""
        self_words = ['i', 'me', 'my', 'mine', 'myself']
        words = text.lower().split()
        if not words:
            return 0
        self_count = sum(1 for word in words if word in self_words)
        return self_count / len(words)

    def _detect_negative_cognition(self, text):
        """Detect negative cognitive patterns"""
        negative_patterns = [
            r"i can't", r"i won't", r"i'll never", r"it's impossible",
            r"i always", r"i never", r"everything is", r"nothing works"
        ]

        count = 0
        text_lower = text.lower()
        for pattern in negative_patterns:
            if re.search(pattern, text_lower):
                count += 1

        return count

    def _detect_positive_outlook(self, text):
        """Detect positive outlook patterns"""
        positive_patterns = [
            r"i can", r"i will", r"i'm able", r"it's possible",
            r"looking forward", r"excited about", r"happy to"
        ]

        count = 0
        text_lower = text.lower()
        for pattern in positive_patterns:
            if re.search(pattern, text_lower):
                count += 1

        return count

    def _detect_uncertainty(self, text):
        """Detect uncertainty and hesitation"""
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'unsure']
        words = text.lower().split()
        if not words:
            return 0
        uncertainty_count = sum(1 for word in words if word in uncertainty_words)
        return uncertainty_count / len(words)

    def extract_emotional_keywords(self, text):
        """Extract keywords with emotional significance"""
        if not text or self.nlp is None:
            return []

        try:
            doc = self.nlp(text)
            emotional_keywords = []

            for token in doc:
                if (token.pos_ in ['ADJ', 'ADV', 'VERB'] and
                        not token.is_stop and
                        len(token.text) > 2):
                    emotional_keywords.append({
                        'word': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'sentiment': self._get_word_sentiment(token.text)
                    })

            return emotional_keywords
        except Exception as e:
            print(f"Emotional keywords extraction error: {e}")
            return []

    def _get_word_sentiment(self, word):
        """Get basic sentiment for individual word"""
        positive_words = {'good', 'great', 'excellent', 'happy', 'joy', 'love', 'nice', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'sad', 'hate', 'angry', 'horrible', 'hate'}

        word_lower = word.lower()
        if word_lower in positive_words:
            return 'positive'
        elif word_lower in negative_words:
            return 'negative'
        else:
            return 'neutral'

    def calculate_readability_metrics(self, text):
        """Calculate readability and complexity metrics"""
        if not text:
            return {
                'readability_score': 0,
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'word_count': 0,
                'sentence_count': 0
            }

        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)

            if not sentences or not words:
                return {
                    'readability_score': 0,
                    'avg_sentence_length': 0,
                    'avg_word_length': 0,
                    'word_count': 0,
                    'sentence_count': 0
                }

            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)

            # Simple readability score (higher = more complex)
            readability_score = (avg_sentence_length + avg_word_length) / 2

            return {
                'readability_score': readability_score,
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'word_count': len(words),
                'sentence_count': len(sentences)
            }
        except Exception as e:
            print(f"Readability metrics error: {e}")
            return {
                'readability_score': 0,
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'word_count': 0,
                'sentence_count': 0
            }