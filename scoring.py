import spacy
from spellchecker import SpellChecker
import language_tool_python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans
nlp = spacy.load("en_core_web_sm")
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk




def guess_topic(essay):
    # Tokenize the essay into words
    words = word_tokenize(essay)
    # Filter out stopwords
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
    # Count word frequencies
    word_freq = Counter(filtered_words)
    # Get the most common words
    common_words = word_freq.most_common(5)  # Adjust the number to get more or fewer topics
    return [word for word, freq in common_words]


def misspelledWords(essay, initial_score):
    spell = SpellChecker()
    doc = nlp(essay)
    words = [token.text for token in doc if token.is_alpha]
    misspelled = spell.unknown(words)

    # Adjust the score based on the number of misspelled words
    if len(misspelled) > 5:
        score = 0
    elif len(misspelled) > 3:
        score = 1
    elif len(misspelled) > 1:
        score = 2
    elif len(misspelled) > 0:
        score = 3
    else:
        score = 4

    return misspelled, score


def essayLength(essay):
    
    doc = nlp(essay)
    sentences = list(doc.sents)
    complexSentences = sum(1 for sentence in sentences if sum(1 for token in sentence if token.pos == "VERB") > 1)
    if complexSentences < 2:
        feedback = "Very few complex sentences detected; essay is too short."
        score = 0
    elif complexSentences < 4:
        feedback = "Essay is somewhat short."
        score = 1
    elif complexSentences < 6:
        feedback = "Moderate length."
        score = 2
    elif complexSentences < 8:
        feedback = "Good length."
        score = 3
    elif complexSentences < 10:
        feedback = "Very good length."
        score = 4
    else:
        feedback = "Excellent length, very complex structure."
        score = 5
    return feedback, score


def checkGrammar(essay, initial_score):
    # Initialize grammar checking tool
    tool = language_tool_python.LanguageTool('en-US')
    # Check the essay for grammatical issues
    matches = tool.check(essay)
    grammar_issues = ["{0} (Context: {1}) [Rule ID: {2}]".format(match.message, match.context, match.ruleId) for match in matches]
    # Calculate score deductions for grammar issues
    deductions_per_issue = 3
    score = max(0, initial_score - len(grammar_issues) * deductions_per_issue)
    return grammar_issues, score


def parse_essay(essay):
    
    # Segment the essay into sentences
    sentences = sent_tokenize(essay)
    
    # List to hold the parse trees
    parse_trees = []
    
    # Process each sentence
    for sentence in sentences:
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        # Tag the tokens with parts of speech
        tagged_tokens = pos_tag(tokens)
        # Generate the parse tree
        tree = ne_chunk(tagged_tokens)
        parse_trees.append(tree)
    
    return parse_trees


def check_sentence_starts(essay):
    doc = nlp(essay)
    issues = []
    for sent in doc.sents:
        first_token = sent[0]
        if first_token.tag_ in ['WDT', 'WP', 'WP$', 'WRB']:  # WH-words
            if sent[-1].text != '?':
                issues.append(f"Question should end with a question mark: '{sent.text}'")
        elif first_token.tag_.startswith('VB'):  # Verbs
            if first_token.tag_ not in ['VBG', 'VBN']:  # Gerunds or past participles might be okay in certain contexts
                issues.append(f"Declarative sentence starts with a verb: '{sent.text}'")
        elif first_token.pos_ == 'AUX':
            if sent[-1].text != '?':
                issues.append(f"Question should end with a question mark: '{sent.text}'")
    return issues


def check_missing_constituents(essay):
    doc = nlp(essay)
    issues = []
    for sent in doc.sents:
        # Check for missing determiners before singular nouns
        for token in sent:
            if token.dep == 'det' and token.head.pos == 'NOUN' and token.head.tag == 'NN':
                if token.text.lower() not in ['a', 'an', 'the']:
                    issues.append(f"Missing determiner before singular noun: '{token.head.text}' in '{sent.text}'")
        # More rules can be added here based on specific needs
    return issues


def check_subordinating_conjunctions(essay):
    doc = nlp(essay)
    issues = []
    for sent in doc.sents:
        for token in sent:
            if token.dep == "mark":  # 'mark' dependency tag is used for subordinating conjunctions
                if token.text.lower() in ['because', 'if', 'when', 'although']:
                    # Check for the presence of a finite verb in the clause
                    clause = list(token.head.subtree)
                    has_finite_verb = any(t.tag.startswith('VB') and t.dep != 'gerund' for t in clause)
                    has_gerund = any(t.dep == 'gerund' for t in clause)

                    if token.text.lower() == 'because' and has_gerund:
                        issues.append(f"Error with 'because': gerund used instead of a finite verb in '{sent.text}'")
                    elif token.text.lower() != 'because' and not has_finite_verb and not has_gerund:
                        issues.append(f"Error with '{token.text}': No finite verb or gerund in '{sent.text}'")

    return issues


def get_word_embeddings(essay):
    
    # Process the essay
    doc = nlp(essay)
    
    # Extract embeddings for each token that has a vector
    word_vectors = [(token.text, token.vector) for token in doc if token.has_vector]
    
    # Calculate the average vector for the essay, excluding tokens that don't have vectors
    average_vector = np.mean([token.vector for token in doc if token.has_vector], axis=0)
    
    return word_vectors, average_vector


def essay_to_vec(essay, dimensions=300):  # Default to 300, which is typical for GloVe
    
    # Analyze the essay
    doc = nlp(essay)

    # Filter content words and compute their vectors
    vectors = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and token.has_vector:
            vectors.append(token.vector[:dimensions])

    # Calculate the average of the vectors
    if vectors:
        average_vector = np.mean(vectors, axis=0)
    else:
        average_vector = np.zeros(dimensions)

    return average_vector


def extract_keywords(text, num_keywords=5):
    """
    Extract keywords from the text using spaCy for POS tagging and lemmatization.
    """
    doc = nlp(text)
    # Filter tokens that are common words or punctuation marks
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    word_freq = Counter(words)
    # Return the most common keywords
    keywords = [word for word, freq in word_freq.most_common(num_keywords)]
    return keywords


def formulate_prompt(keywords):
    
    # Check if there are enough keywords to form a prompt
    if not keywords:
        return "Discuss the key aspects of your topic."
    
    # Create a topic sentence from keywords
    topic_sentence = ' and '.join([', '.join(keywords[:-1]), keywords[-1]] if len(keywords) > 1 else keywords)
    
    # Formulate a complete prompt
    prompt = f"Do you agree or disagree with the following statement? Successful people should focus on {topic_sentence}. Use specific reasons and examples to support your answer."
    
    return prompt


def get_average_vector(text):
    
    doc = nlp(text)
    vectors = [token.vector for token in doc if token.has_vector and not token.is_stop]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros((300,)) 


def cosine_similarity(vec1, vec2):
    
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def analyze_essay(essay):
    
    keywords = extract_keywords(essay)
    hypothetical_prompt = formulate_prompt(keywords)
    prompt_vector = get_average_vector(hypothetical_prompt)
    essay_vector = get_average_vector(essay)
    similarity = cosine_similarity(prompt_vector, essay_vector)
    return hypothetical_prompt, similarity


def sentence_coherence(essay):
    sentences = nltk.sent_tokenize(essay)
    embeddings = [get_embedding(sentence) for sentence in sentences]
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        similarities.append(sim)
    # Analyze variations here
    return np.std(similarities)


# def evaluate_grammar(essay):
#     doc = nlp(essay)
#     issues = []

#     for sent in doc.sents:
#         root = [token for token in sent if token.head == token][0]  # Identify the root of the sentence
#         if root.pos != "VERB" and root.pos != "AUX":
#             issues.append(f"Potential sentence fragment without main verb: '{sent.text}'")

#         for token in sent:
#             # Check for missing subjects in main clauses
#             if token.dep == "ROOT" and not any(child.dep == "nsubj" for child in token.children):
#                 issues.append(f"Missing subject in sentence: '{sent.text}'")
#             # Check for other common errors (e.g., incorrect prepositions, verb forms, etc.)
#             if token.dep == "prep" and not any(child.dep_ == "pobj" for child in token.children):
#                 issues.append(f"Preposition without object: '{token.text}' in '{sent.text}'")

#     return issues





# def evaluate_sentences(essay):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(essay)
#     issues = []

#     for sent in doc.sents:
#         tokens = list(sent)
#         # Check sentence start for declarative and interrogative sentences
#         if tokens[0].tag_ in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:  # Starts with a verb
#             if sent.text.endswith('?'):
#                 if tokens[0].tag_ not in ['VBZ', 'VBP']:  # Not starting with an auxiliary or modal verb
#                     issues.append(f"Question may not start properly with a verb: '{sent.text}'")
#             else:
#                 issues.append(f"Declarative sentence starts with a verb: '{sent.text}'")
        
#         # Check for missing constituents
#         for token in sent:
#             if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ == 'VERB':
#                 if not any(child for child in token.children if child.dep_ in ['det', 'nsubjpass']):
#                     if token.tag_ != 'NNP':  # Proper nouns do not always need a determiner
#                         issues.append(f"Possible missing determiner before '{token.text}' in '{sent.text}'")

#         # Check for proper use of subordinating conjunctions
#         for token in sent:
#             if token.dep_ == 'mark':  # Token is a subordinating conjunction
#                 if token.head.pos_ != 'VERB':
#                     issues.append(f"Subordinating conjunction '{token.text}' not followed by a verb in '{sent.text}'")

#     return issues








# def extract_keywords(text, num_keywords=5):
#     """
#     Extract keywords from the text using spaCy for POS tagging and lemmatization.
#     """
#     doc = nlp(text)
#     # Filter tokens that are common words or punctuation marks
#     words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
#     word_freq = Counter(words)
#     # Return the most common keywords
#     keywords = [word for word, freq in word_freq.most_common(num_keywords)]
#     return keywords


# def formulate_prompt(keywords):
    
#     # Check if there are enough keywords to form a prompt
#     if not keywords:
#         return "Discuss the key aspects of your topic."
    
#     # Create a topic sentence from keywords
#     topic_sentence = ' and '.join([', '.join(keywords[:-1]), keywords[-1]] if len(keywords) > 1 else keywords)
    
#     # Formulate a complete prompt
#     prompt = f"Do you agree or disagree with the following statement? Successful people should focus on {topic_sentence}. Use specific reasons and examples to support your answer."
    
#     return prompt


# def get_average_vector(text):
    
#     doc = nlp(text)
#     vectors = [token.vector for token in doc if token.has_vector and not token.is_stop]
#     if vectors:
#         return np.mean(vectors, axis=0)
#     else:
#         return np.zeros((300,)) 


# def cosine_similarity(vec1, vec2):
    
#     if np.all(vec1 == 0) or np.all(vec2 == 0):
#         return 0.0
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# def analyze_essay(essay):
    
#     keywords = extract_keywords(essay)
#     hypothetical_prompt = formulate_prompt(keywords)
#     prompt_vector = get_average_vector(hypothetical_prompt)
#     essay_vector = get_average_vector(essay)
#     similarity = cosine_similarity(prompt_vector, essay_vector)
#     return hypothetical_prompt, similarity


def finalGrade(grades):
    # Calculate the final grade based on averaged component scores
    total = sum(grades) / len(grades) if grades else 0
    # Categorize the final score
    category = "high" if total >= 60 else "low"
    return total, category