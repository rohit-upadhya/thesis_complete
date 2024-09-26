import os
import fitz  # PyMuPDF
from src.dataset.commons import utils
import csv
import json

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
            # print(type(link))  # For debugging, prints the type of the link
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

        if size == current_size and size >11.5:
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
            # print("text before query ",text)
            while(len(query)>0 and size >= query[-1][1]):
                query.pop()
            query.append([text, size])
        # print("final query after append ",query)
        query_tuple = []
        for item in query:
            query_tuple.append(item[0].split(". ")[-1].strip())
        query_tuple = tuple(query_tuple)
        final_results.append((text, size, font, link, query_tuple))
    return final_results

def filter_out_links_para(results):
    final_results = []
    for result in results:
        if(result[3] is not None and "§" in result[0] and "i=" in result[3]):
            final_results.append(result)
    return final_results

def combine_paragraph_numbers(results):
    combined_results = {}

    for result in results:
        text, size, font, link, query, para_nums = result
        # Convert inner lists to tuples for the key
        query_key = tuple(tuple(item) if isinstance(item, list) else item for item in query)
        key = (query_key, link)
        # Store the original query format and other details
        if key not in combined_results:
            combined_results[key] = {'texts': [], 'para_nums': set(), 'size': size, 'font': font, 'original_query': query}
        combined_results[key]['texts'].append(text)
        combined_results[key]['para_nums'].update(para_nums)
    
    final_results = []
    for key, value in combined_results.items():
        query_key, link = key
        combined_text = ' '.join(value['texts'])
        size = value['size']
        font = value['font']
        para_nums = sorted(value['para_nums'])
        # Convert query back to list of lists
        original_query = [list(item) if isinstance(item, tuple) else item for item in value['original_query']]
        original_query = tuple(original_query)
        final_results.append((combined_text, size, font, link, original_query, para_nums))
    
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
    docs  = utils.get_mongo_docs()
    # docs = list(docs)
    # print(docs[:10])
    final_results = []
    for result in results:
        text, size, font, link, query, para_nums = result
        paragraphs = []
        id = link.split("i=")[1]
        case_heading = utils.capture_case_heading(id, docs)
        for paragraph_number in para_nums:
            # paragraph, html = utils.capture_paragrahs(id, paragraph_number, docs)
            paragraph = utils.capture_paragraphs(id, paragraph_number, docs)
            for item in paragraph:
                paragraphs.append(item)
        # print(len(paragraphs))
        # paragraphs = "\n".join(paragraphs)
        final_results.append((text,size,font,link, query, para_nums, paragraphs, case_heading))
    return final_results


def make_csv(results):
    csv_file_output =  os.path.join("output","english.csv")
    with open(csv_file_output, mode='w', newline='') as file:
        writer = csv.writer(file,delimiter='|')
        writer.writerows(results)  

def convert_to_json(final_result, file_name = "results.json"):
    # Define the output JSON structure
    json_result = []
    for result in final_result:
        entry = {
            "query": result[4],
            "case_name": result[7],
            "paragrpahs": result[6],
            "paragraph_numbers": result[5],
            "link": result[3],
        }
        json_result.append(entry)

    # Write the JSON object to a file
    file_name = os.path.join("output", "english", file_name)
    with open(file_name, "w") as file:
        json.dump(json_result, file, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    raw_data_path = "raw_data/english"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(raw_data_path):
        for filename in filenames:
            if "pdf" in filename:
                files.append(os.path.join(dirpath, filename))
    print(files)
    for file in files :
        file_name = file.split("/")[-1].split(".pdf")[0]
        results = scrape(file)
        filtered_results = filter_results(results=results)
        combined_links = combine_adjacent_entries_with_same_link(results=filtered_results)
        combined_size = combine_adjacent_entries_with_same_size(results=combined_links)
        seperate_links = separate_links(combined_size)
        combined_results = combine_entries_with_section(seperate_links)
        results_with_query = build_query(combined_results)
        relevant_results = filter_out_links_para(results_with_query)
        relevant_results_with_para_num = obtain_paragraph_numbers(relevant_results)
        relevant_results_triplet = combine_paragraph_numbers(relevant_results_with_para_num)
        final_result = obtain_paragraphs(relevant_results_triplet)
        
        # for result in final_results:
        #     text, size, font, link = result
        #     if link is None:
        #         if "§" in text:
        #             print(result)
        # print(relevant_results_triplet[0:3])
        # text_file_output =  os.path.join("output","english","results2.txt")           
        # with open(text_file_output, "w+") as file:
        #     for result in final_result:
        #         # if result[3] in "https://hudoc.echr.coe.int/eng?i=001-105606":
        #         # file.write(f"Query: {result[4]}, Text: {result[0]}, Size: {result[1]}, Font: {result[2]}, Link: {result[3]}\n")
        #         # file.write(f"Text: {result[0]}, Size: {result[1]}, Font: {result[2]}, Link: {result[3]}\n")
        #         file.write(f"Query: {result[4]}, Text: {result[0]}, Para No.: {result[5]}, Size: {result[1]}, Font: {result[2]}, Link: {result[3]}, Paragraph: {result[6]}n")
        #             # file.write(f"Query: {result[4]}, Text: {result[0]}, Para No.: {result[5]}, Size: {result[1]}, Font: {result[2]}, Link: {result[3]}\n")
        #         # file.write("\n")
        # # make_csv(final_result)
        print("number of obtained query, case, paragraph triplets : ",len(final_result))
        convert_to_json(file_name=f"{file_name}.json", final_result=final_result)
        number_result_file_output =  os.path.join("output","english", "english_results.txt")
        with open(number_result_file_output, "a+") as file:
            file.write(f"Number of results in {file_name} = {len(final_result)} \n")
    
    
    
        #         # file.write(f"Text: {result[0]}, Size: {result[1]}, Font: {result[2]}, Link: {result[3]}\n")
        #         # file.write(f"Query: {result[4]}, Text: {result[0]}, Para No.: {result[5]} Size: {result[1]}, Font: {result[2]}, Link: {result[3]}, Paragraph: {result[6]}\n")
        #         # file.write("\n")