import pandas as pd
import numpy as np
import spacy


nlp = spacy.load("en_core_web_sm")

def essay_grader(essay):
    # Tokenize the essay and calculate its length
    tokens = nlp(essay)
    essay_length = len(tokens)

    # Length scoring
    length_score = calculate_length_score(essay_length)
    
    # analysis using SpaCy
    syntactic_score = calculate_syntactic_score(tokens)
    
    # Calculate total score
    total_score = (length_score + syntactic_score) / 2
    
    return {
        'total_score': total_score,
        'length_score': length_score,
        'syntactic_score': syntactic_score,
        'essay_length': essay_length
    }

def calculate_length_score(essay_length):
    
    if essay_length < 200:
        return 0.5
    elif essay_length < 500:
        return 0.7
    else:
        return 1.0

def calculate_syntactic_score(tokens):
   
    num_sentences = len(list(tokens.sents))
    
    
    if num_sentences < 10:
        return 0.5
    
    elif num_sentences < 15:
        return 0.7
    
    else:
        return 1.0

# Example usage
input_essay = "For my part, I do not think that young people enjoy their life more than adults. I have in mind further reasons to explain that children or young people also have some problems. Fisrtly, I think this kind of statement is obviously done by an adult. Why? Because adults or old people have, must of the time, a certain nostalgy of their young years. I mean they tend to have better souvenirs than what really happenned in their life. The truth is not in the memories.Secondly, adults often forget that young people have problems, and sometimes it is not minor problems. They can meet some difficulties at school. If children are not good students they might meet some difficulties to have friends or good relationship with their professors or even with their parents. Even worse, young people are used to face the violence. Parents and adults often bear in mind what was the school 20 years ago, that is to say safe and quiet. It is not the case todayany more. Children face racist and rude words from other students.Thirdly, and it may be the main point, we need to wonder why when you are young you always want to be an adult. I remenber when I was ten or eleven, I wanted to be older to be allowed to watch TV in the evening or to be able to do everything I want to.  We have to bear in mind the childhood and the pre-adult steps are really very hard. Young people must build their own identity, but in the same time they are under influence. According to official figures, suicide is one the main death causes after car accidents. It shows how it is difficult to be young.As a conclusion, it is obvious that yound people heve more leisure times and holidays than adults. But they also meet situation and problem to sort out. To be honest I think people are able to enjoy their like along their all life they just have to do the right choice, be happy or sad."
result = essay_grader(input_essay)
print(result)
