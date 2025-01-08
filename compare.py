if __name__ == "__main__":
    f_original = open('lang-8\l8_train_original.txt', 'r')
    f_corrected = open('lang-8\l8_train_corrected_result.txt', 'r')
    f_references = open('lang-8\l8_train_reference.txt', 'r')
    # x = f_corrected.read()
    # print(len(x))
    # y = f_references.read()
    # print(len(y))

    original = []
    corrected = []

    lines_original_list = []
    lines_corrected_list = []
    lines_references_list = []
    
    
    for lines_original in f_original.readlines():
        original.append(lines_original)
        lines_original = lines_original.replace(" ", "")
        # print(lines_original)
        lines_original_list.append(lines_original)

    for lines_corrected in f_corrected.readlines():
        corrected.append(lines_corrected)
        lines_corrected = lines_corrected.replace(" ", "")
        # print(lines_corrected)
        lines_corrected_list.append(lines_corrected)
    
    # for lines_references in f_references.readlines():
    #     lines_references = lines_references.replace(" ", "")
    #     # print(lines_references)
    #     lines_references_list.append(lines_references)

    line_index_list = []
    for i in range(len(lines_corrected_list)):
        if(lines_original_list[i] != lines_corrected_list):
            # print("original: ", lines_original_list[i])
            # print("corrected: " , lines_corrected_list[i])
            # print(i)
            line_index_list.append(i)
    # print(line_index_list)

    for i in line_index_list:
        print("original: ", original[i])
        print("corrected: " ,corrected[i])


    # for i in range(len(lines_corrected_list)):
    #     if(lines_corrected_list[i] == lines_references_list):
    #         print(lines_corrected_list[i])
    #         print(i)
    
    print("================end=================")