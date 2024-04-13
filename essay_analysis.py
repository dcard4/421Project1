from scoring import essayLength, misspelledWords, checkGrammar, finalGrade

def processEssay(essay_text):
    feedback_length, score_length = essayLength(essay_text)
    misspelled, score_spelling = misspelledWords(essay_text, score_length)
    grammar_issues, score_grammar = checkGrammar(essay_text, score_spelling)

    final_score, category = finalGrade([score_length, score_spelling, score_grammar])
    
    print(f"Essay Analysis Results:")
    print(f"Length Feedback: {feedback_length}")
    print(f"Complex Sentence Score: {score_length}")
    print(f"Spelling Score: {score_spelling} (Misspelled Words: {', '.join(misspelled) if misspelled else 'None'})")
    print(f"Grammar Score: {score_grammar} (Issues: {'; '.join(grammar_issues) if grammar_issues else 'None'})")
    print(f"Total Score: {final_score} ({category})")