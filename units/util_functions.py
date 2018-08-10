import numpy as np
import re


# Since linear sum aasinment implementation only takes square matrices as input,
# distance matrix needs to be transformed to the square shape; values from the
# added rows are not used in further wrangling process
def pad_matrix(a, pad_value=0):
    m = a.reshape((a.shape[0], -1))
    padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
    padded[0:m.shape[0], 0:m.shape[1]] = m
    return padded


# function which extracts conditional FDs from the file where they are stored in the form of plain strings
def extract_cfds(cfd_list):
    extracted_cfds = []
    for i in range(0, cfd_list.shape[0]):
        # print(i)
        cfd = str(cfd_list.iloc[i][0])

        # extract attribute names
        regex = r'\[[\w\s\,]+\]'
        reg = re.compile(regex)
        res = reg.findall(cfd)
        left_attr = res[0]
        right_attr = res[1]

        regex = r'[\w\d]+[\w\d\s]*'
        reg = re.compile(regex)
        left_attr = reg.findall(left_attr)
        right_attr = reg.findall(right_attr)


        # extract left values
        regex = r'[(][\w\d]+.*[|]'
        reg = re.compile(regex)
        left_val = reg.findall(cfd)

        regex = r'[\w\d]+[\w\d\s]*'
        reg = re.compile(regex)
        left_val_list = reg.findall(left_val[0])


        # extract right values
        regex = r'[|][\w\d]+.*[)]'
        reg = re.compile(regex)
        right_val = reg.findall(cfd)

        regex = r'[\w\d]+[\w\d\s]*'
        reg = re.compile(regex)
        right_val_list = reg.findall(right_val[0])

        # create left and right parts of the rule
        lhs = {}
        rhs = {}

        # add attribute names and values to the rule
        for counter in range(0, len(left_attr)):
            lhs[left_attr[counter]] = left_val_list[counter]

        for counter in range(0, len(right_attr)):
            rhs[right_attr[counter]] = right_val_list[counter]

        # compose CFD in a form of dictionary
        new_cfd = {}
        new_cfd['lhs'] = lhs
        new_cfd['rhs'] = rhs

        # append composed CFD to the list
        extracted_cfds.append(new_cfd)

    return extracted_cfds
