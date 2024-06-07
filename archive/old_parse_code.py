import pandas as pd
import os
import numpy as np
from PyPDF2 import PdfReader
import re
import sys
from operator import itemgetter
import fitz
import json
import string

def scrape(keyword, filePath):
    results = [] # list of tuples that store the information as (text, font size, font name) 
    text = []
    pdf = fitz.open(filePath) # filePath is a string that contains the path to the pdf
    for page in pdf:
        dict = page.get_text("dict")
        blocks = dict["blocks"]
        for block in blocks:
            if "lines" in block.keys():
                spans = block['lines']
                # print(block)
                # print(spans)
                for span in spans:
                    data = span['spans']
                    # print(data)
                    for lines in data:
                        # if keyword in lines['text'].lower(): # only store font information of a specific keyword]
                            # print(lines)
                            # return
                            results.append((lines['text'], lines['size'], lines['font']))
                            text.append(lines['text'].links())
                            # lines['text'] -> string, lines['size'] -> font size, lines['font'] -> font name
    pdf.close()
    return results, text

if __name__=="__main__":
    raw_data_path = "raw_data"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(raw_data_path):
        for filename in filenames: 
            if "pdf" in filename:
                files.append(os.path.join(dirpath, filename))
    print(files)
    results, text = scrape('a',files[0])
    with open("text.txt", "w+") as file:
        file.write(str(text))


# this will be the correct one in case the one after does not work.
def scrape(keyword, filePath):
    results = []  # list of tuples that store the information as (text, font size, font name, link)
    pdf = fitz.open(filePath)  # filePath is a string that contains the path to the pdf

    for page_num, page in enumerate(pdf):
        # Extract text and its properties
        dict = page.get_text("dict")
        blocks = dict["blocks"]
        links = page.get_links()  # Get all links on the page

        for block in blocks:
            if "lines" in block.keys():
                spans = block['lines']
                for span in spans:
                    data = span['spans']
                    for lines in data:
                        link = None
                        text_rect = fitz.Rect(lines['bbox'])
                        for link_info in links:
                            link_rect = fitz.Rect(link_info['from'])
                            if link_rect.intersects(text_rect) and link_info.get('uri'):
                                link = link_info['uri']
                                break
                        results.append((lines['text'], lines['size'], lines['font'], link))
    
    pdf.close()
    return results
