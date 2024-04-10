import spacy
from spellchecker import SpellChecker
import language_tool_python


nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()
tool = language_tool_python.LanguageTool('en-US')



def essay_length(essay):
    
    doc = nlp(essay)
    sentences = list(doc.sents)
   
    complexSentences = 0
    for sentence in sentences:
        verbs = [token for token in sentence if token.pos_ == "VERB"]
        if len(verbs) > 1:  
            complexSentences += 1
    
    if complexSentences < 5:
        feedback = f"The essay is too short, only {complexSentences} sentences detected."
        grade = 1
    elif 5 <= complexSentences < 7:
        feedback = f"The essay is somewhat short, with {complexSentences} sentences."
        grade = 2
    elif 7 <= complexSentences < 9:
        feedback = f"The essay is nearing adequate length with {complexSentences} sentences."
        grade = 3
    elif 9 <= complexSentences < 10:
        feedback = f"The essay is almost at the ideal length with {complexSentences} sentences."
        grade = 4
    else:  
        feedback = f"The essay achieves the ideal length with {complexSentences} sentences or more."
        grade = 5


    if complexSentences > 0:
        feedback += f"It has {complexSentences} sentences."

    return feedback, grade
         
    


def misspelled_words_and_grade(essay, initial_grade):
    spell = SpellChecker()
    doc = nlp(essay)
    words = [token.text for token in doc if token.is_alpha]
    
    misspelled = spell.unknown(words)
    
    
    misspelled_count = len(misspelled)
    
    
    grade = initial_grade  
    
    if misspelled_count > 20:
        grade = max(1, grade - 4)  
    elif misspelled_count > 15:
        grade = max(1, grade - 3)
    elif misspelled_count > 10:
        grade = max(1, grade - 2)
    elif misspelled_count > 5:
        grade = max(1, grade - 1)
    

    return list(misspelled), grade



def read_essay_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        essay = file.read()
    return essay


def check_grammar(essay,grade):
    
    tool = language_tool_python.LanguageTool('en-US')
    
   
    matches = tool.check(essay)
    grammar_issues = []
    
    for match in matches:
        issue = f"Error: {match.message}\nContext: {match.context}\nRule ID: {match.ruleId}\n"
        grammar_issues.append(issue)

    
    doc = nlp(essay)
    for sent in doc.sents:
        
        has_subject = any(token.dep_ == 'nsubj' for token in sent)
        has_complex_dependencies = any(token.dep_ not in ['nsubj', 'dobj', 'pobj'] for token in sent)
        
        if not has_subject:
            grammar_issues.append(f"Missing subject in sentence: '{sent}'\n")
        if has_complex_dependencies:
            grammar_issues.append(f"Complex sentence structure detected: '{sent}'\n")

    
    grade = 5  
    issue_count = len(grammar_issues)

    if issue_count > 20:
        grade = 1
    elif issue_count > 15:
        grade = 2
    elif issue_count > 10:
        grade = 3
    elif issue_count > 5:
        grade = 4
    

    return grammar_issues, grade



def calculate_final_grade(grades):
    
    if grades:
        final_grade = sum(grades) / len(grades)
    else:
        final_grade = 0  # Default to 0 if no grades are provided
    return final_grade


if __name__ == "__main__":
    grade = 0   
    file_path = '38209.txt'  
    essay_text = read_essay_from_file(file_path)
    
    feedback, grade = essay_length(essay_text)
    print(feedback)
    print(f"Initial Grade: {grade}")
    
    misspelled, grade = misspelled_words_and_grade(essay_text, grade)
    if misspelled:
        print("Misspelled words found:", ', '.join(misspelled))
    else:
        print("No misspelled words found.")
    
    grammar_issues, grade_grammar = check_grammar(essay_text)  
    if grammar_issues:
        print("Grammar issues found:")
        for issue in grammar_issues:
            print(issue)
    else:
        print("No grammar issues found.")
    
    # Calculate the final grade
    all_grades = [grade_length, grade_spelling, grade_grammar]
    final_grade = calculate_final_grade(all_grades)
    print(f"Final Grade: {final_grade}%")

    
