import spacy
from spellchecker import SpellChecker
import language_tool_python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter



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


# Load the spaCy English model for linguistic analysis
nlp = spacy.load("en_core_web_sm")


def misspelledWords(essay, initial_score):
    spell = SpellChecker()
    nlp = spacy.load("en_core_web_sm")  # Ensure spaCy is loaded inside the function if it's not globally loaded
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
    nlp = spacy.load("encore_web_sm")  # Ensure spaCy is loaded inside the function if it's not globally loaded
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


def evaluate_grammar(essay):
    doc = nlp(essay)
    issues = []

    for sent in doc.sents:
        root = [token for token in sent if token.head == token][0]  # Identify the root of the sentence
        if root.pos != "VERB" and root.pos != "AUX":
            issues.append(f"Potential sentence fragment without main verb: '{sent.text}'")

        for token in sent:
            # Check for missing subjects in main clauses
            if token.dep == "ROOT" and not any(child.dep == "nsubj" for child in token.children):
                issues.append(f"Missing subject in sentence: '{sent.text}'")
            # Check for other common errors (e.g., incorrect prepositions, verb forms, etc.)
            if token.dep == "prep" and not any(child.dep_ == "pobj" for child in token.children):
                issues.append(f"Preposition without object: '{token.text}' in '{sent.text}'")

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


def check_sentence_starts(essay):
    doc = nlp(essay)
    issues = []
    for sent in doc.sents:
        first_token = sent[0]
        if first_token.tag in ['WDT', 'WP', 'WP$', 'WRB']:  # WH-words
            if sent[-1].text != '?':
                issues.append(f"Question should end with a question mark: '{sent.text}'")
        elif first_token.tag.startswith('VB'):  # Verbs
            if first_token.tag not in ['VBG', 'VBN']:  # Gerunds or past participles might be okay in certain contexts
                issues.append(f"Declarative sentence starts with a verb: '{sent.text}'")
        elif first_token.pos == 'AUX':
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


def finalGrade(grades):
    # Calculate the final grade based on averaged component scores
    total = sum(grades) / len(grades) if grades else 0
    # Categorize the final score
    category = "high" if total >= 60 else "low"
    return total, category



def parse_tree(essay):
    # Tokenize the essay text
    tokens = nltk.word_tokenize(essay)
    # Tag the tokens with parts of speech
    tagged_tokens = nltk.pos_tag(tokens)
    # Generate the parse tree using named entity chunking
    tree = nltk.chunk.ne_chunk(tagged_tokens)
    return tree



