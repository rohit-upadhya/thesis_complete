from pymongo import MongoClient
import re
import pandas as pd
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_mongo_docs():
    # db connection setup
    URI = "mongodb://%s:%s@f27se1.in.tum.de:27017/echr" % ("echr_read", "echr_read")# local
    # URI = "mongodb://%s:%s@localhost:27017/echr" % ("echr_read", "echr_read") # server
    client = MongoClient(URI)
    database = client['echr']
    hejud = database["hejud"]
    # db setup
    # all = database["merged_scrape"]
    
    # getting docs
    # docs = all.find({'doctype': 'HEJUD'})
    return hejud

def extract_paragraph_numbers(text):
    """
    Extract numbers following the § symbol from the given text.
    Handles both single numbers and ranges (e.g., §§ 26-32).
    """
    # paragraph_numbers = []

    # Regular expression to find patterns like § 26 or §§ 26-32
    pattern = re.compile(r'§{1,2}\s*(\d+)(?:-(\d+))?')
    match = pattern.search(text)
    if match:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start

        # Return all numbers in the range as a list
        return list(range(start, end + 1))

    return []
    # pattern = re.compile(r'§{1,2}\s*(\d+)(?:-(\d+))?')

    # matches = pattern.findall(text)
    # for match in matches:
    #     start = int(match[0])
    #     end = int(match[1]) if match[1] else start

    #     # Add all numbers in the range to the list
    #     paragraph_numbers.extend(range(start, end + 1))

    # return paragraph_numbers

def extract_paragraphs_from_sentences(id, paragraph, docs):
    
    document = docs.find_one({'_id': id})
    # for key in document.keys():
    #         print(key)
    # docs = docs.aggregate([{ '$sample': { 'size': 2 } }])
    paragraphs = []
    try:
        if document is None:
            return []
        
        sentences = document["sentences"]
        text_file_output =  os.path.join("output","english","sentences.txt")   
        with open(text_file_output, "w+") as file:
            for result in sentences:
                file.write(f"{result} \n\n")
        print(id)
        flag = 0
        for doc in sentences:
            if f"{paragraph}. " in doc:
                flag = 1
            elif f"{paragraph+1}. " in doc and flag==1:
                break
            if flag == 1:
                paragraphs.append(doc)
        return paragraphs
    except Exception as e:
        print(f"Paragraph not present for this id {id}")
        return []
        
    
    # text_file_output =  os.path.join("output","db.txt")  
    # with open(text_file_output, "w+") as file:
    #     for result in docs:
    #         file.write(f" ***** {len(result)}  ***** {result} \n\n")

def truncate_to_one_decimal(num):
    return int(num * 10) / 10.0

def compare(num1, num2):
    return truncate_to_one_decimal(num1) == truncate_to_one_decimal(num2)

def extract_paragraph_from_html(id, paragraph, docs):
    document = docs.find_one({'_id': id})
    paragraphs = []
    try:
        if document is None:
            print("document not present")
            return []
        html = document["html"]
        flag = 0
        if not html:
            print("No HTML content found")
            return []
        paragraphs = extract_paragraphs_within_class(html, "s30EEC3F8", str(paragraph))
             
        return paragraphs
    except Exception as e:
        print(f"Paragraph not present for this id : {id}")
        return []
    
def extract_paragraphs_within_class(html_content, target_class, target_string):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')

    results = []
    capture = False
    text_file_output =  os.path.join("output","english","html.txt")   
    with open(text_file_output, "w+") as file:
            file.write(f"{html_content} \n\n")
    for p in paragraphs:
        text = p.get_text()
        p_class = p.get('class', [])
        if f"{target_string}." in text and target_class in p_class:
            capture = True
        if capture:
            results.append(text)
            next_sibling = p.find_next_sibling('p')
            if next_sibling and target_class in next_sibling.get('class', []):
                break

    return results

def find_overlapping_paragraphs(paragraphs1, paragraphs2, threshold=0.7):
    try:
        vectorizer = TfidfVectorizer().fit_transform(paragraphs1 + paragraphs2)
        vectors = vectorizer.toarray()
        similarity_matrix = cosine_similarity(vectors[:len(paragraphs1)], vectors[len(paragraphs1):])
        
        overlapping_pairs = []
        for i in range(len(paragraphs1)):
            for j in range(len(paragraphs2)):
                if similarity_matrix[i, j] > threshold:
                    overlapping_pairs.append((paragraphs1[i], paragraphs2[j], similarity_matrix[i, j]))
        
        return overlapping_pairs
    except Exception as e:
        print("issue with similarity check. Ignoring this datapoint")
        return None

def capture_paragraphs(id, paragraph_no, docs):
    html_paragraph = extract_paragraph_from_html(id, paragraph_no, docs)
    sentence_paragraphs = extract_paragraphs_from_sentences(id, paragraph_no, docs)
    # print("html_paragraph : ",html_paragraph)
    # print("sentence_paragraphs : ",sentence_paragraphs)
    if len(html_paragraph) == 0:
        return sentence_paragraphs
    if len(sentence_paragraphs) == 0:
        return html_paragraph
    threshold = 0.85
    overlapping_paragraphs = find_overlapping_paragraphs(html_paragraph, sentence_paragraphs, threshold)
    paragraphs = []
    if overlapping_paragraphs is None:
        return []
    for extracted, additional, similarity in overlapping_paragraphs:
        paragraphs.append(additional)
    return sentence_paragraphs

def capture_case_heading(id, docs):
    try:
        document = docs.find_one({'_id': id})
        return document['docname']
    except Exception as e:
        print(f"no doc present for id : {id}")
        return ""

if __name__ == "__main__":
    docs  = get_mongo_docs()
    paragraph = extract_paragraph_from_html("001-105606", 136, docs)
    additional_paragraphs = extract_paragraphs_from_sentences("001-105606", 136, docs)[0]
    
    threshold = 0.85  # Adjust this value as needed
    print("going to extracted now")
    overlapping_paragraphs = find_overlapping_paragraphs(paragraph, additional_paragraphs, threshold)
    
    for extracted, additional, similarity in overlapping_paragraphs:
        print("Extracted Paragraph:", extracted)
        print("Overlapping Additional Paragraph:", additional)
        print("Similarity:", similarity)
        print("\n---\n")
    print(paragraph)
    