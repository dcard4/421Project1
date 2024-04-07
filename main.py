import spacy
from spellchecker import SpellChecker
import language_tool_python


nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()
tool = language_tool_python.LanguageTool('en-US')



def essay_length(essay):
    doc = nlp(essay)
    numSentences = len(list(doc.sents))

    if numSentences <= 6:
        feedback = f"The essay is not long enough, only {numSentences} sentences detected."
        grade = 2
    elif 7 <= numSentences <= 9:
        feedback = f"The essay is somewhat short, with {numSentences} sentences."
        grade = 3
    elif numSentences == 10:
        feedback = f"The essay is adequately long with {numSentences} sentences."
        grade = 4
    elif numSentences >= 14:
        feedback = f"The essay is sufficiently long with {numSentences} sentences."
        grade = 5
    else:
        feedback = "The essay's length is unconventional."
        grade = 3

    return feedback, grade 
         
    


def misspelledWords(essay,grade):
    
    doc = nlp(essay)
    words = [token.text for token in doc if token.is_alpha]  
    
    misspelled = spell.unknown(words)
    
    return list(misspelled)



def read_essay_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        essay = file.read()
    return essay


def check_grammar(essay):
    matches = tool.check(essay)
    grammarIssues = []
    for match in matches:
        issue = f"{match.ruleIssueType}: {match.message} - {match.context}"
        grammarIssues.append(issue)
    return grammarIssues




if __name__ == "__main__":
    grade = 0   
    file_path = '38209.txt'  
    essay_text = read_essay_from_file(file_path)
    feedback, grade = essay_length(essay_text)
    print(feedback)
    print(f"Grade: {grade}")
    



    
# grade = 0
    
# file_path = '38209.txt'  


# essay_text = read_essay_from_file(file_path)


# result = essay_length(essay_text, grade)
# print(result)

    # misspelledWords = misspelledWords(essay_text)
    # if misspelledWords:
    #     print("Misspelled words found:", ', '.join(misspelledWords))
    # else:
    #     print("No misspelled words found.")
        
        
    # grammarIssues = check_grammar(essay_text)
    # if grammarIssues:
    #     print("Grammar issues found:")
    #     for issue in grammarIssues:
    #         print(issue)
    # else:
    #     print("No grammar issues found.")    
