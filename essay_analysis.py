from scoring import essayLength, misspelledWords, checkGrammar, finalGrade

def processEssay(essay_text):
    # Evaluate the essay's length based on complex sentence count
    # Returns feedback on essay length and a numerical score for this aspect
    feedback_length, score_length = essayLength(essay_text)

    # Check for spelling errors in the essay and adjust the length score based on findings
    # Returns a list of misspelled words and a new, adjusted score accounting for spelling errors
    misspelled, score_spelling = misspelledWords(essay_text, score_length)

    # Assess the essay for grammar issues using the score adjusted for spelling errors
    # Returns a list of grammar issues and a further adjusted score based on these issues
    grammar_issues, score_grammar = checkGrammar(essay_text, score_spelling)

    # Calculate the final score of the essay by averaging the scores from length, spelling, and grammar assessments
    # Also categorizes the final score into a qualitative category ("high" or "low")
    final_score, category = finalGrade([score_length, score_spelling, score_grammar])
    
    # Print out the results of the essay analysis
    # Includes feedback and scores for each aspect, as well as the final aggregated score and category
    print(f"Essay Analysis Results:")
    print(f"Length Feedback: {feedback_length}")
    print(f"Complex Sentence Score: {score_length}")
    print(f"Spelling Score: {score_spelling} (Misspelled Words: {', '.join(misspelled) if misspelled else 'None'})")
    print(f"Grammar Score: {score_grammar} (Issues: {'; '.join(grammar_issues) if grammar_issues else 'None'})")
    print(f"Total Score: {final_score} ({category})")
