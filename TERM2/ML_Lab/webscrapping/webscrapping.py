from bs4 import BeautifulSoup

with open('webscrp.html','r') as html_file:
    content = html_file.read()
    # print(content)

    soup = BeautifulSoup(content,'lxml')
    
    # find all h5 tags in the file 
    tags = soup.find_all('h5')
    # print(tags)
    
    #prints all tags 
    for course in tags:
        print(course.text)
        