from essay_analysis import processEssay


# Check if the script is being run as the main program
if __name__ == "__main__":
    # Prompt the user for the file path
    file_path = input("Please enter the path to your essay file: ")

    with open(file_path, 'r') as file:
        essay_text = file.read()

    processEssay(essay_text)
