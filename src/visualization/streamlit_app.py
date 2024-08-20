import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Base path for JSON files
BASE_PATH = "/home/upadro/code/thesis/output"
RESULTS_PATH = "/home/upadro/code/thesis/output/visualizations"
ANALYTICS_PATH = "/home/upadro/code/thesis/data_analysis"
LANGUAGES = [ "italian", "romanian", "russian", "turkish", "ukrainian", "french", "english"]

# Set the default language
DEFAULT_LANGUAGE = "italian"

# Function to save computation result to a JSON file
def save_computation(data, is_satisfactory, language):
    data['is_satisfactory'] = is_satisfactory
    results_ = []
    results_file_path = os.path.join(RESULTS_PATH, f"{language}_results.json")
    try:
        with open(results_file_path, 'r') as f:
            results_json = json.load(f)
        for result in results_json:
            results_.append(result)
    except json.JSONDecodeError:
        print(f"Error: The file is blank or contains invalid JSON.")
    except FileNotFoundError:
        print(f"Error: The file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    results_.append(data)
    with open(results_file_path, 'w') as f:
        json.dump(results_, f, indent=4)
    return results_file_path

# Function to list all JSON files in a directory
def list_json_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.json')]

# Function to calculate the number of satisfactory and unsatisfactory results
def calculate_satisfactory_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    true_count = sum(1 for item in data if item.get('is_satisfactory') == True)
    false_count = sum(1 for item in data if item.get('is_satisfactory') == False)
    
    return true_count, false_count

# Function to load JSON data
def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main"])

if page == "Main":
    tab1, tab2, tab3 = st.tabs(["JSON Checker", "Results Calculation", "Analytics"])

    with tab1:
        st.title("JSON Satisfactory Checker")
        language = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_checker")

        if language:
            foldername = os.path.join(BASE_PATH, language, "analysis")
            if os.path.exists(foldername):
                json_files = list_json_files(foldername)
                
                if json_files:
                    filename = st.selectbox("Select a JSON file", json_files, key="file_checker")
                    filepath = os.path.join(foldername, filename)
                    
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as file:
                            json_data = json.load(file)
                        
                        if isinstance(json_data, list):
                            if 'page' not in st.session_state:
                                st.session_state.page = 0

                            col1, col2 = st.columns([3, 1])

                            with col1:
                                current_data = json_data[st.session_state.page]
                                st.write(f"Displaying JSON object {st.session_state.page + 1} of {len(json_data)}:", current_data)

                            with col2:
                                is_satisfactory = st.checkbox("Is the JSON satisfactory?", key=f"checkbox_{st.session_state.page}")

                                if st.button("Save Computation"):
                                    save_filename = save_computation(current_data, is_satisfactory, language)
                                    st.success(f"Computation saved to {save_filename}")

                                if st.session_state.page < len(json_data) - 1:
                                    if st.button("Next"):
                                        st.session_state.page += 1
                                        st.experimental_rerun()
                                
                                if st.session_state.page > 0:
                                    if st.button("Previous"):
                                        st.session_state.page -= 1
                                        st.experimental_rerun()

                        else:
                            st.error("The provided JSON file is not a list of objects.")
                    else:
                        st.error("File not found. Please check the filename and try again.")
                else:
                    st.error("No JSON files found in the specified folder.")
            else:
                st.error("Folder not found. Please check the folder name and try again.")

    with tab2:
        st.title("Results Calculation")
        language = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_calculation")

        if language:
            foldername = RESULTS_PATH
            results_file_path = os.path.join(foldername, f"{language}_results.json")
            
            if os.path.exists(results_file_path):
                true_count, false_count = calculate_satisfactory_results(results_file_path)
                
                st.write(f"Number of satisfactory results: {true_count}")
                st.write(f"Number of unsatisfactory results: {false_count}")
            else:
                st.error("Results file not found. Please check the filename and try again.")

    with tab3:
        st.title("Analytics")
        language = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_analytics")

        if language:
            filename = f"{language}.json"
            filepath = os.path.join(ANALYTICS_PATH, filename)

            if os.path.exists(filepath):
                data = load_json_data(filepath)

                if isinstance(data, list):
                    percentages = []
                    query_tokens = []

                    # Extract data from file_meta_data_information
                    for item in data:
                        if 'file_meta_data_information' in item:
                            for meta_data in item['file_meta_data_information']:
                                percentages.append(meta_data.get('percentage', 0) * 100)  # Convert to percentage
                                query_tokens.append(meta_data.get('query_tokens', 0))

                    # Calculate mean and median
                    mean_percentage = np.mean(percentages)
                    median_percentage = np.median(percentages)
                    mean_tokens = np.mean(query_tokens)
                    median_tokens = np.median(query_tokens)

                    # Count occurrences of each value for percentages and query tokens
                    percentage_counts = Counter(percentages)
                    query_tokens_counts = Counter(query_tokens)

                    # Plotting the percentage distribution (discrete values)
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.bar(percentage_counts.keys(), percentage_counts.values(), color='salmon', width=0.2)
                    ax1.axvline(mean_percentage, color='black', linestyle='--', label='Mean')
                    ax1.axvline(median_percentage, color='orange', linestyle='--', label='Median')
                    ax1.set_xlim(-2, max(percentage_counts.keys()) + 5)  # Adjust x-axis limits to focus on the relevant data
                    ax1.set_ylim(0, max(percentage_counts.values()) + 5)  # Scale y-axis appropriately
                    ax1.set_title('% of relevant paragraphs per judgement')
                    ax1.set_xlabel('Percentage')
                    ax1.set_ylabel('Frequency')
                    ax1.legend()
                    st.pyplot(fig1)

                    # Plotting the query tokens distribution (discrete values)
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.bar(query_tokens_counts.keys(), query_tokens_counts.values(), color='salmon', width=1)
                    ax2.axvline(mean_tokens, color='black', linestyle='--', label='Mean')
                    ax2.axvline(median_tokens, color='orange', linestyle='--', label='Median')
                    ax2.set_xlim(-2, max(query_tokens_counts.keys()) + 2)  # Adjust x-axis limits to add space before 0 and at the end
                    ax2.set_ylim(0, max(query_tokens_counts.values()) + 10)  # Scale y-axis appropriately
                    ax2.set_title('Number of tokens per query')
                    ax2.set_xlabel('Query Tokens')
                    ax2.set_ylabel('Frequency')
                    ax2.legend()
                    st.pyplot(fig2)

                else:
                    st.error("The selected JSON file does not contain a list of data points.")
            else:
                st.error("File not found. Please check the filename and try again.")
