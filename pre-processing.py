def preprocess(filename, shakespeare=True):
    file = open(filename, "r")

    if shakespeare:
        poetry_list_by_line = shakespeare_extract_poetry_list(file)
    else:
        poetry_list_by_line = spencer_extract_poetry_list(file)

    poetry_list_by_line = tokenize_punctuation(poetry_list_by_line)

    print(poetry_list_by_line[1])

def tokenize_punctuation(poetry_list_by_line):
    punctuations = "!#$%&()*+,./:;<=>?@[\]^_{|}~"
    for i in range(len(poetry_list_by_line)):
        for punctuation in punctuations:
            poetry_list_by_line[i] = poetry_list_by_line[i].replace(punctuation,  f" {punctuation} ")

    return poetry_list_by_line

# Return a list of poems, each poem is a string
def shakespeare_extract_poetry_list(file):
    text = file.read()
    poetry_list = text.lower().split("\n\n")

    for i in range(len(poetry_list)):
        start_line = 2 if i != 0 else 1
        lines = poetry_list[i].split("\n")
        poetry_list[i] = ' NEWLINE '.join(lines[start_line:])

    return poetry_list

def spencer_extract_poetry_list(file):
    pass

preprocess("data/shakespeare.txt")
#
# test = "when forty winters shall besiege thy brow, NEWLINE and dig deep trenches in thy "
# test = test.replace(',', " , ")
# print(test)