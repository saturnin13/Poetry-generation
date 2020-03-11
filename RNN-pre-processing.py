def extract_characters(n_jump, size_input, shakespeare=True):
    file = open("data/shakespeare.txt", "r") if shakespeare else open("data/spenser.txt", "r")

    if shakespeare:
        poetry_list_by_line = shakespeare_extract_poetry_list(file)
    else:
        poetry_list_by_line = spencer_extract_poetry_list(file)

    result = split_to_fixed_size_list(n_jump, poetry_list_by_line, size_input)

    return result


def split_to_fixed_size_list(n_jump, poetry_list_by_line, size_input):
    result = []

    for poem in poetry_list_by_line:
        poem = remove_leading_line_spaces(poem)

        for i in range(int(len(poem) / n_jump)):
            curr_pos_start = i * n_jump

            if curr_pos_start + size_input >= len(poem):
                break

            result.append(poem[curr_pos_start:curr_pos_start + size_input])

    return result

def remove_leading_line_spaces(poem):
    while "  " in poem:
        poem = poem.replace("  ", "")

    return poem

def shakespeare_extract_poetry_list(file):
    text = file.read()
    poetry_list = text.lower().split("\n\n")

    for i in range(len(poetry_list)):
        start_line = 2 if i != 0 else 1
        lines = poetry_list[i].split("\n")
        poetry_list[i] = '\n'.join(lines[start_line:])

    return poetry_list


def spencer_extract_poetry_list(file):
    text = file.read()
    poetry_list = text.lower().split("\n\n")

    for i in range(len(poetry_list) - 1, -1, -1):
        if i % 2 == 0:
            poetry_list.remove(poetry_list[i])

        else:
            pass

    return poetry_list


print(extract_characters(8, 40, False))
