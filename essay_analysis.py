from scoring import essayLength, misspelledWords, checkGrammar, finalGrade, guess_topic,parse_essay,check_sentence_starts,check_missing_constituents
# ,evaluate_sentences,check_subordinating_conjunctions
from scoring import essay_to_vec,check_subordinating_conjunctions,get_word_embeddings,analyze_essay,sentence_coherence



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

    # Guess the main topics of the essay
    guessed_topics = guess_topic(essay_text)

    # Calculate the final score of the essay by averaging the scores from length, spelling, and grammar assessments
    # Also categorizes the final score into a qualitative category ("high" or "low")
    final_score, category = finalGrade([score_length, score_spelling, score_grammar])
    
    trees = parse_essay(essay_text)
    
    sentence_start_issues = check_sentence_starts(essay_text)
    constituent_issues = check_missing_constituents(essay_text)
    conjunction_issues = check_subordinating_conjunctions(essay_text)
    word_vectors, avg_vector = get_word_embeddings(essay_text)
    essay_vector = essay_to_vec(essay_text)
    prompt, similarity_score = analyze_essay(essay_text)
    coherence_score = sentence_coherence(essay_text)
    
    # prompt, similarity_score = analyze_essay(essay_text)
    

   
    

    # Print out the results of the essay analysis
    # Includes feedback and scores for each aspect, as well as the final aggregated score and category
    print(f"Essay Analysis Results:")
    print(f"Length Feedback: {feedback_length}")
    print(f"Complex Sentence Score: {score_length}")
    print(f"Spelling Score: {score_spelling} (Misspelled Words: {', '.join(misspelled) if misspelled else 'None'})")
    print(f"Grammar Score: {score_grammar} (Issues: {'; '.join(grammar_issues) if grammar_issues else 'None'})")
    print(f"Guessed Topics: {', '.join(guessed_topics)}")  # Display guessed topics
    print(f"Sentence Start Issues: {'; '.join(sentence_start_issues) if sentence_start_issues else 'None'}")
    print(f"Constituent Issues: {'; '.join(constituent_issues) if constituent_issues else 'None'}")
    print(f"Subordinating Conjunction Issues: {'; '.join(conjunction_issues) if conjunction_issues else 'None'}")
    
    print("Parse Tree:")
    for tree in trees:
        print(tree)
        
    print("Word Embeddings:")
    for word, vector in word_vectors:
        print(f"Word: {word}, Vector: {vector[:5]}...")  # Display first 5 elements for brevity
    
    print("Average Vector of the Essay:", avg_vector[:5])  # Display first 5 elements for brevity
    print("Essay Vector:", essay_vector)
    print("Hypothetical Prompt:", prompt)
    print("Cosine Similarity:", similarity_score)
    print("Coherence Score:", coherence_score)
    
    print(f"Total Score: {final_score} ({category})")
    print(f"Total Score: {final_score} ({category})")
    
