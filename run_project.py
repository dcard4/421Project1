from essay_analysis import processEssay

# Check if the script is being run as the main program
if __name__ == "__main__":
    print("Please paste your essay and type 'END' on a new line when you are finished:")

    # Initialize an empty list to collect the essay's paragraphs
    essay_lines = []
    while True:
        line = input()
        if line == "END":  # Check if the user has typed the termination string
            break
        essay_lines.append(line)

    # Join all the collected lines into a single string, preserving line breaks
    essay_text = "\n".join(essay_lines)

    # Process the essay text through the function imported from the essay_analysis module
    processEssay(essay_text)
