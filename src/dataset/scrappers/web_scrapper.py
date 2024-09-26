import requests
from bs4 import BeautifulSoup
import urllib.request
import urllib.error

# def web_scrapper(url, paragraph_number):
#     # Fetch the webpage content
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"Failed to retrieve webpage: {response.status_code}")
#         return None
    
#     # Parse the HTML content using BeautifulSoup
#     soup = BeautifulSoup(response.content, 'html.parser')
#     print(soup)
#     paragraphs = soup.find_all('p')
#     print(paragraphs)
#     for para in paragraphs:
#         if f"§ {paragraph_number}" in para.text:
#             return para.get_text()
#     print(paragraphs)
#     # # Find the paragraph with the specified number
#     # paragraphs = soup.find_all(text=lambda text: text and f"§ {paragraph_number}" in text)

#     # if not paragraphs:
#     #     print(f"Paragraph § {paragraph_number} not found.")
#     #     return None

#     # # Extract the paragraph content
#     # for para in paragraphs:
#     #     parent = para.find_parent('p')
#     #     if parent:
#     #         return parent.get_text()

#     # print(f"Paragraph § {paragraph_number} not found in <p> tags.")
#     # return None
    

# if __name__=="__main__":
    
#     # a = web_scrapper("https://hudoc.echr.coe.int/eng?i=001-211972", 97)
#     a = web_scrapper("""https://hudoc.echr.coe.int/eng#{%22languageisocode%22:[%22ENG%22],%22appno%22:[%2220914/07%22],%22documentcollectionid2%22:[%22CHAMBER%22],%22itemid%22:[%22001-211972%22]}""", 97)
#     print(a)
    

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time

def fetch_webpage_content(url):
    # Specify the path to your ChromeDriver
    chrome_driver_path = '/path/to/chromedriver/chromedriver'  # Adjust the path as necessary
    service = Service(chrome_driver_path)
    
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome(service=service)

    # Navigate to the URL
    driver.get(url)

    # Wait for the page to fully load (you can adjust the sleep time if necessary)
    time.sleep(5)

    # Get the page source and close the browser
    html_content = driver.page_source
    driver.quit()
    
    return html_content

def find_paragraphs_recursively(element, paragraph_number):
    matching_paragraphs = []
    if element.name == 'p' and f"§ {paragraph_number}" in element.get_text():
        matching_paragraphs.append(element.get_text())
    
    # Recursively search through all children
    for child in element.find_all(recursive=False):
        matching_paragraphs.extend(find_paragraphs_recursively(child, paragraph_number))
    
    return matching_paragraphs

def find_paragraph_by_number(html_content, paragraph_number):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Start searching from the root of the document
    matching_paragraphs = find_paragraphs_recursively(soup, paragraph_number)
    
    return matching_paragraphs

if __name__ == "__main__":
    url = "https://hudoc.echr.coe.int/eng?i=001-211972"  # URL of the webpage
    paragraph_number = 97  # The paragraph number we are looking for

    try:
        # Fetch and parse the HTML content
        html_content = fetch_webpage_content(url)
        paragraphs = find_paragraph_by_number(html_content, paragraph_number)

        if paragraphs:
            for i, paragraph_text in enumerate(paragraphs, 1):
                print(f"Paragraph § {paragraph_number} - Match {i}:\n{paragraph_text}\n")
        else:
            print(f"Paragraph § {paragraph_number} not found.")
    except Exception as e:
        print(f"Error: {e}")
