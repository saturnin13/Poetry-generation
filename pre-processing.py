import nltk
nltk.download('punkt')

def preprocess(shakespeare=True):
    file = open("data/shakespeare.txt", "r") if shakespeare else open("data/spenser.txt", "r")

    if shakespeare:
        poetry_list_by_line = shakespeare_extract_poetry_list(file)
    else:
        poetry_list_by_line = spencer_extract_poetry_list(file)

    # poetry_list_by_line = tokenize_punctuation(poetry_list_by_line)

    print(poetry_list_by_line[1])

# def tokenize_punctuation(poetry_list_by_line):
#     punctuations = "!#$%&()*+,./:;<=>?@[\]^_{|}~"
#     for i in range(len(poetry_list_by_line)):
#         for punctuation in punctuations:
#             poetry_list_by_line[i] = poetry_list_by_line[i].replace(punctuation,  f" {punctuation} ")
#
#     return poetry_list_by_line

# Return a list of poems, each poem is a string
def shakespeare_extract_poetry_list(file):
    text = file.read()
    poetry_list = text.lower().split("\n\n")

    for i in range(len(poetry_list)):
        start_line = 2 if i != 0 else 1
        lines = poetry_list[i].split("\n")
        poetry_list[i] = nltk.word_tokenize(' NEWLINE '.join(lines[start_line:]))

    return poetry_list

def spencer_extract_poetry_list(file):
    text = file.read()
    poetry_list = text.lower().split("\n\n")

    for i in range(len(poetry_list) - 1, -1, -1):
        if i % 2 == 0:
            poetry_list.remove(poetry_list[i])

        else:
            lines = poetry_list[i].split("\n")
            poetry_list[i] = nltk.word_tokenize(' NEWLINE '.join(lines))

    return poetry_list

preprocess(False)