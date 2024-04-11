import spacy
from spellchecker import SpellChecker
import language_tool_python

nlp = spacy.load("en_core_web_sm")

def essayLength(essay):
    doc = nlp(essay)
    sentences = list(doc.sents)
    complexSentences = sum(1 for sentence in sentences if sum(1 for token in sentence if token.pos_ == "VERB") > 1)
    
    feedback, score = "", 0
    if complexSentences < 5:
        feedback = f"The essay is too short, only {complexSentences} complex sentences detected."
        score = 20  
    elif complexSentences < 7:
        feedback = f"The essay is somewhat short, with {complexSentences} complex sentences."
        score = 40
    elif complexSentences < 9:
        feedback = f"The essay is nearing adequate length with {complexSentences} complex sentences."
        score = 60
    elif complexSentences < 10:
        feedback = f"The essay is almost at the ideal length with {complexSentences} complex sentences."
        score = 80
    else:
        feedback = f"The essay achieves the ideal length with {complexSentences} complex sentences or more."
        score = 100
    return feedback, score



def misspelledWords(essay, initial_score):
    spell = SpellChecker()
    doc = nlp(essay)
    words = [token.text for token in doc if token.is_alpha]
    misspelled = spell.unknown(words)
    
    deductions_per_misspelled = 5  # Deduct 5 points per misspelled word
    score = max(0, initial_score - len(misspelled) * deductions_per_misspelled)
    return list(misspelled), score


def checkGrammar(essay, initial_score):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(essay)
    grammar_issues = [f"Error: {match.message}\nContext: {match.context}\nRule ID: {match.ruleId}\n" for match in matches]
    
    deductions_per_issue = 3  # Deduct 5 points per grammar issue
    score = max(0, initial_score - len(grammar_issues) * deductions_per_issue)
    return grammar_issues, score


def finalGrade(grades):
    # Assuming equal weight for each section; adjust if necessary
    return sum(grades) / len(grades) if grades else 0


if __name__ == "__main__":
    file_path = '38209.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        essay_text = file.read()
    
    feedback, score_length = essayLength(essay_text)
    print(feedback)
    
    misspelled, score_spelling = misspelledWords(essay_text, score_length)
    print("Misspelled words found:", ', '.join(misspelled) if misspelled else "No misspelled words found.")
    
    grammar_issues, score_grammar = checkGrammar(essay_text, score_spelling)
    print("Grammar issues found:\n" + "\n".join(grammar_issues) if grammar_issues else "No grammar issues found.")
    
    final_score = finalGrade([score_length, score_spelling, score_grammar])
    print(f"Final Grade: {final_score}%")
