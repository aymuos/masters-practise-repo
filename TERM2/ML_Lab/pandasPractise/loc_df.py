# understanding loc usage in pandas 

# loc is an accessor in dataframes

import pandas as pd

# Creating a sample DataFrame of student information
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [20, 22, 21, 19, 23],
    'grade': ['A', 'B', 'A', 'C', 'B'],
    'score': [95, 82, 91, 75, 88]
}

df = pd.DataFrame(data)
# print(df.head())

# simple dataframe 

# getting 1st student's information 

student = df.loc[0]  # accesses just 1 row 

# print(student)

# accessing info for second and 4th 

students2=df.loc[[1,3]]
# print(students2)

# accessing specific information for specific rows 

# what if i want to get the name and score for a few people 

details2 = df.loc[[2,3],['name','score']]

# print(details2)

# now we want to combine multiple conditions 

details3 = df.loc[df['score']>90]

print(f'Student scoring more than 90 is : {details3}')

# loc is position inclusive
