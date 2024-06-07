import os
import fitz  # PyMuPDF
from src.commons import utils
import csv

def normalize_link(link):
    if link and link.startswith('http:'):
        return 'https:' + link[5:]
    return link

def scrape(filePath):
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
                        text = lines['text']
                        size = lines['size']
                        font = lines['font']
                        bbox = fitz.Rect(lines['bbox'])
                        line_links = []

                        for link_info in links:
                            link_rect = fitz.Rect(link_info['from'])
                            if link_rect.intersects(bbox) and link_info.get('uri'):
                                line_links.append(link_info['uri'])

                        if line_links:
                            for link in line_links:
                                results.append((text, size, font, link))
                        else:
                            results.append((text, size, font, None))
    
    pdf.close()
    return results


def filter_results(results):
    filtered_results = []
    for result in results:
        text, size, font, link = result
        if size >= 12.0 or "Calibri-Bold" in font or link is not None or "§" in text:
            filtered_results.append(result)
    final_results = []
    for result in filtered_results:
        text, size, font, link = result
        if (size >= 12.0 or "Calibri-Bold" in font) and link is not None:
            final_results.append((text, size, font, None))
        else:
            final_results.append(result)
    return final_results



def combine_adjacent_entries_with_same_link(results):
    combined_results = []
    if not results:
        return combined_results

    combined_text = results[0][0]
    current_size = results[0][1]
    current_font = results[0][2]
    current_link = normalize_link(results[0][3])

    for i in range(1, len(results)):
        text, size, font, link = results[i]
        link = normalize_link(link)
        if link == current_link and current_link is not None:
            if text not in combined_text:
                combined_text += " " + text
        else:
            combined_results.append((combined_text, current_size, current_font, current_link))
            combined_text = text
            current_size = size
            current_font = font
            current_link = link
    
    # Append the last combined result
    combined_results.append((combined_text, current_size, current_font, current_link))
    
    return combined_results
def remove_arial(results):
    final_results = []
    for result in results:
        if "Arial-BoldMT" not in result[2]:
            final_results.append(result)
    return final_results

def remove_comma(results):
    final_results = []
    for result in results:
        if not (len(result[0]) > 1 or result[3] is not None):
            print(result)
        if (len(result[0]) > 1) or result[3] is not None:
            
            final_results.append(result)
    return final_results

def combine_adjacent_entries_with_same_size(results):
    combined_results = []
    if not results:
        return combined_results

    combined_text = results[0][0]
    current_size = results[0][1]
    current_font = results[0][2]
    current_link = results[0][3]

    for i in range(1, len(results)):
        text, size, font, link = results[i]

        if size > 11.5 and utils.compare(size, current_size): #size == current_size and size >11.5:
            combined_text += " " + text
        else:
            # if current_size >11.5:
            combined_results.append((combined_text, current_size, current_font, current_link))
            combined_text = text
            current_size = size
            current_font = font
            current_link = link
    
    # Append the last combined result
    combined_results.append((combined_text, current_size, current_font, current_link))
    
    return combined_results

def separate_links(results):
    final_results = []
    i = 0
    for result in results:
        text, size, font, link = result
        text = text.strip(" ")
        text = text.strip(";")
        if(link is not None and ";" in text):
            texts = text.split(";")
            new_texts = ""
            final_results.append((texts[0], size, font, None))
            for entry in texts[1:]:
                new_texts += entry
            final_results.append((new_texts, size, font, link))
        else:
            final_results.append((text, size, font, link))
    return final_results

def combine_entries_with_section(results):
    final_results = []
    i = 0
    while i < len(results):
        text, size, font, link = results[i]
        if link is not None and i + 1 < len(results) and "§" in results[i + 1][0] and results[i + 1][3] is None:
            combined_text = text + " " + results[i + 1][0]
            final_results.append((combined_text, size, font, link))
            i += 2  # Skip the next entry as it has been combined
        else:
            final_results.append((text, size, font, link))
            i += 1
    return final_results

def build_query(results):
    query = []
    final_results = []
    for result in results:
        text, size, font, link = result
        if size > 11.5:
            while(len(query)>0 and size >= query[-1][1]):
                query.pop()
            query.append([text, size])
        query_tuple = tuple(query)
        final_results.append((text, size, font, link, query_tuple))
    return final_results

def filter_out_links_para(results):
    final_results = []
    for result in results:
        if(result[3] is not None and "§" in result[0] and "i=" in result[3]):
            final_results.append(result)
    return final_results

def obtain_paragraph_numbers(results):
    final_result = []
    for result in results:
        text, size, font, link, query = result
        numbers = utils.extract_paragraph_numbers(text)
        numbers = set(numbers)
        numbers = list(numbers)
        if (len(numbers)>1) and (numbers[0] == numbers[1]):
            numbers = [numbers[0]]
        final_result.append((text,size,font,link, query, numbers))
    return final_result

def obtain_paragraphs(results):
    docs  = utils.capture_paragraph()
    final_results = []
    for result in results:
        text, size, font, link, query, para_nums = result
        paragraphs = []
        id = link.split("i=")[1]
        for paragraph_number in para_nums:
            paragraph = utils.capture_paragrahs(id, paragraph_number, docs)
            paragraphs.append(paragraph)
        final_results.append((text,size,font,link, query, para_nums, paragraphs))
    return final_results


def make_csv(results):
    csv_file_output =  os.path.join("output","english.csv")
    with open(csv_file_output, mode='w', newline='') as file:
        writer = csv.writer(file,delimiter='|')
        writer.writerows(results)  

if __name__ == "__main__":
    raw_data_path = "raw_data/turkish"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(raw_data_path):
        for filename in filenames:
            if "pdf" in filename:
                files.append(os.path.join(dirpath, filename))
    
    print(files)
    results = scrape(files[0])
    filtered_results = filter_results(results=results)
    combined_links = combine_adjacent_entries_with_same_link(results=filtered_results)
    removed_arial = remove_arial(combined_links)
    remove_commas = remove_comma(removed_arial)
    combined_size = combine_adjacent_entries_with_same_size(results=remove_commas)
    seperate_links = separate_links(combined_size)
    combined_results = combine_entries_with_section(seperate_links)
    results_with_query = build_query(combined_results)
    relevant_results = filter_out_links_para(results_with_query)
    relevant_results_with_para_num = obtain_paragraph_numbers(relevant_results)
    # final_result = obtain_paragraphs(relevant_results_with_para_num)
    
    # for result in final_results:
    #     text, size, font, link = result
    #     if link is None:
    #         if "§" in text:
    #             print(result)
    text_file_output =  os.path.join("output","turkish_results.txt")           
    with open(text_file_output, "w+") as file:
        for result in relevant_results_with_para_num:
            file.write(f"Query: {result[4]}, Text: {result[0]}, Para No.: {result[5]}, Size: {result[1]}, Font: {result[2]}, Link: {result[3]}\n")
            # file.write(f"Text: {result[0]}, Size: {result[1]}, Font: {result[2]}, Link: {result[3]}\n")
            # file.write(f"Query: {result[4]}, Text: {result[0]}, Para No.: {result[5]} Size: {result[1]}, Font: {result[2]}, Link: {result[3]}, Paragraph: {result[6]}\n")
            # file.write("\n")
    # make_csv(final_result)