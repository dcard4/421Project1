import spacy
from spellchecker import SpellChecker
import language_tool_python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


nltk.download('punkt')
nltk.download('stopwords')

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

def essayLength(essay):
    # Analyze the essay to break it into sentences
    doc = nlp(essay)
    sentences = list(doc.sents)
    # Count complex sentences based on the number of verbs
    complexSentences = sum(1 for sentence in sentences if sum(1 for token in sentence if token.pos_ == "VERB") > 1)
    
    # Determine feedback and score based on the count of complex sentences
    feedback, score = "", 0
    if complexSentences < 5:
        feedback = "The essay is too short, only {0} complex sentences detected.".format(complexSentences)
        score = 20
    elif complexSentences < 7:
        feedback = "The essay is somewhat short, with {0} complex sentences.".format(complexSentences)
        score = 40
    elif complexSentences < 9:
        feedback = "The essay is nearing adequate length with {0} complex sentences.".format(complexSentences)
        score = 60
    elif complexSentences < 10:
        feedback = "The essay is almost at the ideal length with {0} complex sentences.".format(complexSentences)
        score = 80
    else:
        feedback = "The essay achieves the ideal length with {0} complex sentences or more.".format(complexSentences)
        score = 100
    return feedback, score

def misspelledWords(essay, initial_score):
    # Initialize spell checker
    spell = SpellChecker()
    # Extract words from the essay
    doc = nlp(essay)
    words = [token.text for token in doc if token.is_alpha]
    # Identify misspelled words
    misspelled = spell.unknown(words)
    # Calculate score deductions for misspelled words
    deductions_per_misspelled = 5
    score = max(0, initial_score - len(misspelled) * deductions_per_misspelled)
    return misspelled, score

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

def finalGrade(grades):
    # Calculate the final grade based on averaged component scores
    total = sum(grades) / len(grades) if grades else 0
    # Categorize the final score
    category = "high" if total >= 60 else "low"
    return total, category
