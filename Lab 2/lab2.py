import pandas as pd
import re

titanic = pd.read_csv('Lab 2/titanic.csv')


def get_input():

    print('Search for Name,Age,Sex')
    col = input('Enter column to be searched: ')

    if col not in ['Name', 'Age', 'Sex']:
        print('Invalid Column')
        return

    return match_input(col)


def match_input(col):

    pattern = input('Enter the pattern to match: ')
    output = []

    for data in titanic[col]:

        if re.match(pattern, str(data)):
            output.append(data)

    return output


while True:

    print(get_input())

    search = input("Do you want to search more(Y/N)")

    if search == 'Y':
        get_input()
    else:
        exit(0)
