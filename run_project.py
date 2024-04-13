from essay_analysis import processEssay

# Check if the script is being run as the main program
if __name__ == "__main__":
    # Prompt the user to input their essay text
    essay_text = input("Please enter your essay: ")
    
    # Process the essay text through the function imported from the essay_analysis module
    processEssay(essay_text)
