import nltk
from nltk.parse import CoreNLPParser

# Parser setup
parser = CoreNLPParser(url='http://localhost:9000')

# Example sentence
sentence = "My dog with a broken leg I not want"

# Generate parse tree
parse_tree = next(parser.raw_parse(sentence))

# Function to analyze parse tree for errors
def analyze_tree(tree):
    errors = []
    # Example check for finite verb
    if not any(node.label() == 'VP' and 'VBD' or 'VBZ' in [child.label() for child in node] for node in tree.subtrees()):
        errors.append("Missing finite verb")
    return errors

# Analyze and print errors
errors = analyze_tree(parse_tree)
print("Errors:", errors)
