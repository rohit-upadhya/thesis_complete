import streamlit as st # type: ignore
import json
import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from collections import Counter

# Base path for JSON files
BASE_PATH = "/home/upadro/code/thesis/output/dataset_outputs"
RESULTS_PATH = "/home/upadro/code/thesis/output/visualizations"
ANALYTICS_PATH = "/home/upadro/code/thesis/data_analysis/specifics"
COUNTS_PATH = "/home/upadro/code/thesis/data_analysis/counts"
LANGUAGES = ["all","italian", "romanian", "russian", "turkish", "ukrainian", "french", "english"]
UNSEEN_QUERIES_TRAIN_ANALYTICS_PATH = "/srv/upadro/data_analysis/unseen_queries/train/specifics" 
UNSEEN_QUERIES_TRAIN_COUNTS_PATH = "/srv/upadro/data_analysis/unseen_queries/train/counts"
UNSEEN_QUERIES_TEST_ANALYTICS_PATH = "/srv/upadro/data_analysis/unseen_queries/test/specifics" 
UNSEEN_QUERIES_TEST_COUNTS_PATH = "/srv/upadro/data_analysis/unseen_queries/test/counts"
UNSEEN_QUERIES_VAL_ANALYTICS_PATH = "/srv/upadro/data_analysis/unseen_queries/val/specifics" 
UNSEEN_QUERIES_VAL_COUNTS_PATH = "/srv/upadro/data_analysis/unseen_queries/val/counts"
UNSEEN_QUERIES_UNIQUE_QUERY_ANALYTICS_PATH = "/srv/upadro/data_analysis/unseen_queries/unique_query_test/specifics" 
UNSEEN_QUERIES_UNIQUE_QUERY_COUNTS_PATH = "/srv/upadro/data_analysis/unseen_queries/unique_query_test/counts"
DEFAULT_LANGUAGE = "all"

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

def list_json_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.json')]

def calculate_satisfactory_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    true_count = sum(1 for item in data if item.get('is_satisfactory') == True)
    false_count = sum(1 for item in data if item.get('is_satisfactory') == False)
    
    return true_count, false_count

def load_json_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main", "Train-Test(Unseen Queries)"], index=0)

if page == "Main":
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Analytics"
    tab1, tab2, tab3 = st.tabs(["JSON Checker", "Results Calculation", "Analytics"])

    with tab1:
        st.title("JSON Satisfactory Checker")
        language = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_checker")

        if language:
            foldername = os.path.join(BASE_PATH, language, "unique_query_test")
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
        st.write("_Tokens here refer to individual words (separated by spaces)._")
        language = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_analytics")
        threshold = 1000
        if language:
            filename = f"{language}.json"
            filepath = os.path.join(ANALYTICS_PATH, filename)

            if os.path.exists(filepath):
                data = load_json_data(filepath)

                if isinstance(data, list):
                    percentages = []
                    query_tokens = []
                    relevant_paragraphs_tokens = []
                    total_paragraphs_tokens = []
                    unique_cases = set()
                    total_paragraphs = []
                    for item in data:
                        if 'file_meta_data_information' in item:
                            for meta_data in item['file_meta_data_information']:
                                percentages.append(meta_data.get('percentage', 0) * 100)
                                query_tokens.append(meta_data.get('query_tokens', 0))

                                relevant_paragraphs_tokens.extend(meta_data.get('relevant_paragraphs_tokens', []))
                                total_paragraphs.append(meta_data.get('total_paragraphs', 0))

                                case_link = meta_data.get('case_link')
                                
                                if case_link not in unique_cases:
                                    threshold = 1000
                                    tokens = meta_data.get('total_paragraphs_tokens', [])
                                    grouped_tokens = [token if token <= threshold else threshold for token in tokens]
                                    
                                    total_paragraphs_tokens.extend(grouped_tokens)
                                    unique_cases.add(case_link)
                    mean_percentage = np.mean(percentages)
                    median_percentage = np.median(percentages)
                    mean_tokens = np.mean(query_tokens)
                    median_tokens = np.median(query_tokens)
                    mean_relevant_tokens = np.mean(relevant_paragraphs_tokens)
                    median_relevant_tokens = np.median(relevant_paragraphs_tokens)
                    mean_total_tokens = np.mean(total_paragraphs_tokens)
                    median_total_tokens = np.median(total_paragraphs_tokens)

                    percentage_counts = Counter(percentages)
                    query_tokens_counts = Counter(query_tokens)
                    relevant_tokens_counts = Counter(relevant_paragraphs_tokens)
                    total_tokens_counts = Counter(total_paragraphs_tokens)

                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.bar(percentage_counts.keys(), percentage_counts.values(), color='salmon', width=0.2)
                    ax1.axvline(mean_percentage, color='black', linestyle='--', label='Mean')
                    ax1.axvline(median_percentage, color='orange', linestyle='--', label='Median')
                    ax1.set_xlim(-2, max(percentage_counts.keys()) + 5)
                    ax1.set_ylim(0, max(percentage_counts.values()) + 5)
                    ax1.set_title('% of relevant paragraphs per judgement', fontsize=18)
                    ax1.set_xlabel('Percentage', fontsize=16)
                    ax1.set_ylabel('Frequency', fontsize=16)
                    ax1.tick_params(axis='both', which='major', labelsize=14)
                    ax1.legend(fontsize=12)
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.bar(query_tokens_counts.keys(), query_tokens_counts.values(), color='#D8BFD8', width=1)
                    ax2.axvline(mean_tokens, color='black', linestyle='--', label='Mean')
                    ax2.axvline(median_tokens, color='orange', linestyle='--', label='Median')
                    ax2.set_xlim(-2, max(query_tokens_counts.keys()) + 2)
                    ax2.set_ylim(0, max(query_tokens_counts.values()) + 10)
                    ax2.set_title('Number of tokens per query', fontsize=18)
                    ax2.set_xlabel('Query Tokens', fontsize=16)
                    ax2.set_ylabel('Frequency', fontsize=16)
                    ax2.tick_params(axis='both', which='major', labelsize=14)
                    ax2.legend(fontsize=12)
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    ax3.bar(relevant_tokens_counts.keys(), relevant_tokens_counts.values(), color='lightblue', width=10)
                    ax3.axvline(mean_relevant_tokens, color='black', linestyle='--', label='Mean')
                    ax3.axvline(median_relevant_tokens, color='orange', linestyle='--', label='Median')
                    ax3.set_xlim(-200, max(relevant_tokens_counts.keys()) + 20)
                    ax3.set_ylim(0, max(relevant_tokens_counts.values()) + 5)
                    ax3.set_title('Relevant Paragraph Tokens Distribution', fontsize=18)
                    ax3.set_xlabel('Relevant Paragraph Tokens', fontsize=16)
                    ax3.set_ylabel('Frequency', fontsize=16)
                    ax3.tick_params(axis='both', which='major', labelsize=14)
                    ax3.legend(fontsize=12)
                    st.pyplot(fig3)

                    # Plot the histogram
                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    ax4.bar(total_tokens_counts.keys(), total_tokens_counts.values(), color='lightgreen', width=10)
                    ax4.axvline(mean_total_tokens, color='black', linestyle='--', label='Mean')
                    ax4.axvline(median_total_tokens, color='orange', linestyle='--', label='Median')

                    # Set axis limits
                    ax4.set_xlim(-50, 1050)  # Adjusted to give some padding
                    ax4.set_ylim(0, max(total_tokens_counts.values()) + 1000)

                    # Set custom x-ticks and labels
                    ax4.set_xticks([0, 200, 400, 600, 800, 1000])
                    ax4.set_xticklabels(['0', '200', '400', '600', '800', '>1000'], fontsize=12)

                    # Update title and labels
                    ax4.set_title('Total Paragraph Tokens Distribution (Grouped Above 1000)', fontsize=18)
                    ax4.set_xlabel('Total Paragraph Tokens', fontsize=16)
                    ax4.set_ylabel('Frequency', fontsize=16)
                    ax4.tick_params(axis='both', which='major', labelsize=14)
                    ax4.legend(fontsize=12)

                    # Show the plot
                    st.pyplot(fig4)


                    mean_total_paragraphs = np.mean(total_paragraphs)
                    median_total_paragraphs = np.median(total_paragraphs)
                    total_paragraphs_counts = Counter(total_paragraphs)

                    # New plot for total paragraphs distribution
                    fig6, ax6 = plt.subplots(figsize=(8, 6))
                    ax6.bar(total_paragraphs_counts.keys(), total_paragraphs_counts.values(), color='#8FBC8F', width=1)  # Soft mint green
                    ax6.axvline(mean_total_paragraphs, color='black', linestyle='--', label='Mean')
                    ax6.axvline(median_total_paragraphs, color='orange', linestyle='--', label='Median')
                    ax6.set_xlim(0, max(total_paragraphs_counts.keys()) + 20)
                    ax6.set_ylim(0, max(total_paragraphs_counts.values()) + 5)
                    ax6.set_title('Total Paragraphs Distribution per Case', fontsize=18)
                    ax6.set_xlabel('Number of Total Paragraphs', fontsize=16)
                    ax6.set_ylabel('Frequency', fontsize=16)
                    ax6.tick_params(axis='both', which='major', labelsize=14)
                    ax6.legend(fontsize=12)
                    st.pyplot(fig6)
                    
                else:
                    st.error("The selected JSON file does not contain a list of data points.")
            else:
                st.error("File not found. Please check the filename and try again.")
                
        st.title("Query-Doc Pairs and Unique Queries Across All Languages")

        languages_data = []
        for lang in LANGUAGES:
            counts_filepath = os.path.join(COUNTS_PATH, f"{lang}.json")
            if os.path.exists(counts_filepath):
                with open(counts_filepath, 'r') as f:
                    counts_data = json.load(f)
                num_q_d_pairs = counts_data.get("number_of_q_d_pairs", 0)
                num_unique_queries = counts_data.get("number_of_unique_queries", 0)
                languages_data.append((lang.capitalize(), num_q_d_pairs, num_unique_queries))

        labels = [item[0] for item in languages_data]
        q_d_pairs = [item[1] for item in languages_data]
        unique_queries = [item[2] for item in languages_data]

        x = np.arange(len(labels))
        width = 0.35

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        bars1 = ax6.bar(x - width/2, q_d_pairs, width, label='Query-Doc Pairs')
        bars2 = ax6.bar(x + width/2, unique_queries, width, label='Unique Queries')

        ax6.set_xlabel('Languages')
        ax6.set_ylabel('Count')
        ax6.set_title('Query-Doc Pairs and Unique Queries by Language')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.legend()

        def autolabel(bars):
            """Attach a text label above each bar in *bars*, displaying its height."""
            for bar in bars:
                height = bar.get_height()
                ax6.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)

        st.pyplot(fig6)


elif page == "Train-Test(Unseen Queries)":
    # Create two tabs: one for TRAIN and one for TEST
    train_tab, test_tab, val_tab, unique_query_test_tab = st.tabs(["TRAIN", "TEST", "VAL", "UNIQUE QUERY TEST"])

    # Train Tab
    with train_tab:
        st.title("TRAIN Data Analytics")
        st.write("_Tokens here refer to BERT Tokenizer generated tokens._")
        
        # Select language
        language_train = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_train")
        
        if language_train:
            filename = f"{language_train}.json"
            filepath = os.path.join(UNSEEN_QUERIES_TRAIN_ANALYTICS_PATH, filename)  # Use TRAIN_ANALYTICS_PATH for Train tab

            if os.path.exists(filepath):
                data = load_json_data(filepath)

                if isinstance(data, list):
                    percentages = []
                    query_tokens = []
                    relevant_paragraphs_tokens = []
                    total_paragraphs_tokens = []
                    unique_cases = set()
                    total_paragraphs = []

                    for item in data:
                        if 'file_meta_data_information' in item:
                            for meta_data in item['file_meta_data_information']:
                                percentages.append(meta_data.get('percentage', 0) * 100)
                                total_paragraphs.append(meta_data.get('total_paragraphs', 0))
                                query_tokens.append(meta_data.get('query_tokens', 0))
                                relevant_paragraphs_tokens.extend(meta_data.get('relevant_paragraphs_tokens', []))
                                case_link = meta_data.get('case_link')

                                if case_link not in unique_cases:
                                    total_paragraphs_tokens.extend(meta_data.get('total_paragraphs_tokens', []))
                                    unique_cases.add(case_link)

                    # Calculate the statistics
                    mean_percentage = np.mean(percentages)
                    median_percentage = np.median(percentages)
                    mean_tokens = np.mean(query_tokens)
                    median_tokens = np.median(query_tokens)
                    mean_relevant_tokens = np.mean(relevant_paragraphs_tokens)
                    median_relevant_tokens = np.median(relevant_paragraphs_tokens)
                    mean_total_tokens = np.mean(total_paragraphs_tokens)
                    median_total_tokens = np.median(total_paragraphs_tokens)
                    mean_total_paragraphs = np.mean(total_paragraphs)
                    median_total_paragraphs = np.median(total_paragraphs)
                    total_paragraphs_counts = Counter(total_paragraphs)

                    percentage_counts = Counter(percentages)
                    query_tokens_counts = Counter(query_tokens)
                    relevant_tokens_counts = Counter(relevant_paragraphs_tokens)
                    total_tokens_counts = Counter(total_paragraphs_tokens)
                    total_paragraphs_counts = Counter(total_paragraphs)
                    # Plot the relevant graphs
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.bar(percentage_counts.keys(), percentage_counts.values(), color='salmon', width=0.2)
                    ax1.axvline(mean_percentage, color='black', linestyle='--', label='Mean')
                    ax1.axvline(median_percentage, color='orange', linestyle='--', label='Median')
                    ax1.set_xlim(-2, max(percentage_counts.keys()) + 5)
                    ax1.set_ylim(0, max(percentage_counts.values()) + 5)
                    ax1.set_title('% of relevant paragraphs per judgement')
                    ax1.set_xlabel('Percentage')
                    ax1.set_ylabel('Frequency')
                    ax1.legend()
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.bar(query_tokens_counts.keys(), query_tokens_counts.values(), color='#D8BFD8', width=1)
                    ax2.axvline(mean_tokens, color='black', linestyle='--', label='Mean')
                    ax2.axvline(median_tokens, color='orange', linestyle='--', label='Median')
                    ax2.set_xlim(-2, max(query_tokens_counts.keys()) + 2)
                    ax2.set_ylim(0, max(query_tokens_counts.values()) + 10)
                    ax2.set_title('Number of tokens per query')
                    ax2.set_xlabel('Query Tokens')
                    ax2.set_ylabel('Frequency')
                    ax2.legend()
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    ax3.bar(relevant_tokens_counts.keys(), relevant_tokens_counts.values(), color='lightblue', width=10)
                    ax3.axvline(mean_relevant_tokens, color='black', linestyle='--', label='Mean')
                    ax3.axvline(median_relevant_tokens, color='orange', linestyle='--', label='Median')
                    ax3.set_xlim(-200, max(relevant_tokens_counts.keys()) + 20)
                    ax3.set_ylim(0, max(relevant_tokens_counts.values()) + 5)
                    ax3.set_title('Relevant Paragraph Tokens Distribution')
                    ax3.set_xlabel('Relevant Paragraph Tokens')
                    ax3.set_ylabel('Frequency')
                    ax3.legend()
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    ax4.bar(total_tokens_counts.keys(), total_tokens_counts.values(), color='lightgreen', width=10)
                    ax4.axvline(mean_total_tokens, color='black', linestyle='--', label='Mean')
                    ax4.axvline(median_total_tokens, color='orange', linestyle='--', label='Median')
                    ax4.set_xlim(-200, max(total_tokens_counts.keys()) + 20)
                    ax4.set_ylim(0, max(total_tokens_counts.values()) + 1000)
                    ax4.set_title('Total Paragraph Tokens Distribution (Log Scale on Y-Axis)')
                    ax4.set_xlabel('Total Paragraph Tokens')
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                    st.pyplot(fig4)
                    
                    fig6, ax6 = plt.subplots(figsize=(8, 6))
                    ax6.bar(total_paragraphs_counts.keys(), total_paragraphs_counts.values(), color='#8FBC8F', width=1)  # Soft mint green
                    ax6.axvline(mean_total_paragraphs, color='black', linestyle='--', label='Mean')
                    ax6.axvline(median_total_paragraphs, color='orange', linestyle='--', label='Median')
                    ax6.set_xlim(0, max(total_paragraphs_counts.keys()) + 20)
                    ax6.set_ylim(0, max(total_paragraphs_counts.values()) + 5)
                    ax6.set_title('Total Paragraphs Distribution per Case')
                    ax6.set_xlabel('Number of Total Paragraphs')
                    ax6.set_ylabel('Frequency')
                    ax6.legend()
                    st.pyplot(fig6)

                else:
                    st.error("The selected JSON file does not contain a list of data points.")
            else:
                st.error("File not found. Please check the filename and try again.")
                
        st.title("Query-Doc Pairs and Unique Queries Across All Languages")

        languages_data = []
        for lang in LANGUAGES:
            counts_filepath = os.path.join(UNSEEN_QUERIES_TRAIN_COUNTS_PATH, f"{lang}.json")
            if os.path.exists(counts_filepath):
                with open(counts_filepath, 'r') as f:
                    counts_data = json.load(f)
                num_q_d_pairs = counts_data.get("number_of_q_d_pairs", 0)
                num_unique_queries = counts_data.get("number_of_unique_queries", 0)
                languages_data.append((lang.capitalize(), num_q_d_pairs, num_unique_queries))

        labels = [item[0] for item in languages_data]
        q_d_pairs = [item[1] for item in languages_data]
        unique_queries = [item[2] for item in languages_data]

        x = np.arange(len(labels))
        width = 0.35

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        bars1 = ax6.bar(x - width/2, q_d_pairs, width, label='Query-Doc Pairs')
        bars2 = ax6.bar(x + width/2, unique_queries, width, label='Unique Queries')

        ax6.set_xlabel('Languages')
        ax6.set_ylabel('Count')
        ax6.set_title('Query-Doc Pairs and Unique Queries by Language')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.legend()

        def autolabel(bars):
            """Attach a text label above each bar in *bars*, displaying its height."""
            for bar in bars:
                height = bar.get_height()
                ax6.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)

        st.pyplot(fig6)

    # Test Tab
    with test_tab:
        st.title("Test Data Analytics")
        st.write("_Tokens here refer to BERT Tokenizer generated tokens._")
        
        # Select language
        language_test = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_test")
        
        if language_test:
            filename = f"{language_test}.json"
            filepath = os.path.join(UNSEEN_QUERIES_TEST_ANALYTICS_PATH, filename)  # Use TRAIN_ANALYTICS_PATH for Train tab

            if os.path.exists(filepath):
                data = load_json_data(filepath)

                if isinstance(data, list):
                    percentages = []
                    query_tokens = []
                    relevant_paragraphs_tokens = []
                    total_paragraphs_tokens = []
                    unique_cases = set()
                    total_paragraphs = []

                    for item in data:
                        if 'file_meta_data_information' in item:
                            for meta_data in item['file_meta_data_information']:
                                percentages.append(meta_data.get('percentage', 0) * 100)
                                query_tokens.append(meta_data.get('query_tokens', 0))
                                relevant_paragraphs_tokens.extend(meta_data.get('relevant_paragraphs_tokens', []))
                                case_link = meta_data.get('case_link')
                                total_paragraphs.append(meta_data.get('total_paragraphs', 0))
                                if case_link not in unique_cases:
                                    total_paragraphs_tokens.extend(meta_data.get('total_paragraphs_tokens', []))
                                    unique_cases.add(case_link)

                    # Calculate the statistics
                    mean_percentage = np.mean(percentages)
                    median_percentage = np.median(percentages)
                    mean_tokens = np.mean(query_tokens)
                    median_tokens = np.median(query_tokens)
                    mean_relevant_tokens = np.mean(relevant_paragraphs_tokens)
                    median_relevant_tokens = np.median(relevant_paragraphs_tokens)
                    mean_total_tokens = np.mean(total_paragraphs_tokens)
                    median_total_tokens = np.median(total_paragraphs_tokens)

                    percentage_counts = Counter(percentages)
                    query_tokens_counts = Counter(query_tokens)
                    relevant_tokens_counts = Counter(relevant_paragraphs_tokens)
                    total_tokens_counts = Counter(total_paragraphs_tokens)

                    # Plot the relevant graphs
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.bar(percentage_counts.keys(), percentage_counts.values(), color='salmon', width=0.2)
                    ax1.axvline(mean_percentage, color='black', linestyle='--', label='Mean')
                    ax1.axvline(median_percentage, color='orange', linestyle='--', label='Median')
                    ax1.set_xlim(-2, max(percentage_counts.keys()) + 5)
                    ax1.set_ylim(0, max(percentage_counts.values()) + 5)
                    ax1.set_title('% of relevant paragraphs per judgement')
                    ax1.set_xlabel('Percentage')
                    ax1.set_ylabel('Frequency')
                    ax1.legend()
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.bar(query_tokens_counts.keys(), query_tokens_counts.values(), color='#D8BFD8', width=1)
                    ax2.axvline(mean_tokens, color='black', linestyle='--', label='Mean')
                    ax2.axvline(median_tokens, color='orange', linestyle='--', label='Median')
                    ax2.set_xlim(-2, max(query_tokens_counts.keys()) + 2)
                    ax2.set_ylim(0, max(query_tokens_counts.values()) + 10)
                    ax2.set_title('Number of tokens per query')
                    ax2.set_xlabel('Query Tokens')
                    ax2.set_ylabel('Frequency')
                    ax2.legend()
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    ax3.bar(relevant_tokens_counts.keys(), relevant_tokens_counts.values(), color='lightblue', width=10)
                    ax3.axvline(mean_relevant_tokens, color='black', linestyle='--', label='Mean')
                    ax3.axvline(median_relevant_tokens, color='orange', linestyle='--', label='Median')
                    ax3.set_xlim(-200, max(relevant_tokens_counts.keys()) + 20)
                    ax3.set_ylim(0, max(relevant_tokens_counts.values()) + 5)
                    ax3.set_title('Relevant Paragraph Tokens Distribution')
                    ax3.set_xlabel('Relevant Paragraph Tokens')
                    ax3.set_ylabel('Frequency')
                    ax3.legend()
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    ax4.bar(total_tokens_counts.keys(), total_tokens_counts.values(), color='lightgreen', width=10)
                    ax4.axvline(mean_total_tokens, color='black', linestyle='--', label='Mean')
                    ax4.axvline(median_total_tokens, color='orange', linestyle='--', label='Median')
                    ax4.set_xlim(-200, max(total_tokens_counts.keys()) + 20)
                    ax4.set_ylim(0, max(total_tokens_counts.values()) + 10)
                    ax4.set_title('Total Paragraph Tokens Distribution (Log Scale on Y-Axis)')
                    ax4.set_xlabel('Total Paragraph Tokens')
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                    st.pyplot(fig4)
                    
                    mean_total_paragraphs = np.mean(total_paragraphs)
                    median_total_paragraphs = np.median(total_paragraphs)
                    total_paragraphs_counts = Counter(total_paragraphs)

                    # New plot for total paragraphs distribution
                    fig6, ax6 = plt.subplots(figsize=(8, 6))
                    ax6.bar(total_paragraphs_counts.keys(), total_paragraphs_counts.values(), color='#8FBC8F', width=1)  # Soft mint green
                    ax6.axvline(mean_total_paragraphs, color='black', linestyle='--', label='Mean')
                    ax6.axvline(median_total_paragraphs, color='orange', linestyle='--', label='Median')
                    ax6.set_xlim(0, max(total_paragraphs_counts.keys()) + 20)
                    ax6.set_ylim(0, max(total_paragraphs_counts.values()) + 5)
                    ax6.set_title('Total Paragraphs Distribution per Case')
                    ax6.set_xlabel('Number of Total Paragraphs')
                    ax6.set_ylabel('Frequency')
                    ax6.legend()
                    st.pyplot(fig6)

                else:
                    st.error("The selected JSON file does not contain a list of data points.")
            else:
                st.error("File not found. Please check the filename and try again.")
                
        st.title("Query-Doc Pairs and Unique Queries Across All Languages")

        languages_data = []
        for lang in LANGUAGES:
            counts_filepath = os.path.join(UNSEEN_QUERIES_TEST_COUNTS_PATH, f"{lang}.json")
            if os.path.exists(counts_filepath):
                with open(counts_filepath, 'r') as f:
                    counts_data = json.load(f)
                num_q_d_pairs = counts_data.get("number_of_q_d_pairs", 0)
                num_unique_queries = counts_data.get("number_of_unique_queries", 0)
                languages_data.append((lang.capitalize(), num_q_d_pairs, num_unique_queries))

        labels = [item[0] for item in languages_data]
        q_d_pairs = [item[1] for item in languages_data]
        unique_queries = [item[2] for item in languages_data]

        x = np.arange(len(labels))
        width = 0.35

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        bars1 = ax6.bar(x - width/2, q_d_pairs, width, label='Query-Doc Pairs')
        bars2 = ax6.bar(x + width/2, unique_queries, width, label='Unique Queries')

        ax6.set_xlabel('Languages')
        ax6.set_ylabel('Count')
        ax6.set_title('Query-Doc Pairs and Unique Queries by Language')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.legend()

        def autolabel(bars):
            """Attach a text label above each bar in *bars*, displaying its height."""
            for bar in bars:
                height = bar.get_height()
                ax6.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)

        st.pyplot(fig6)

    with val_tab:
        st.title("Val Data Analytics")
        st.write("_Tokens here refer to BERT Tokenizer generated tokens._")
        
        # Select language
        language_val = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_val")
        
        if language_val:
            filename = f"{language_test}.json"
            filepath = os.path.join(UNSEEN_QUERIES_VAL_ANALYTICS_PATH, filename)  # Use TRAIN_ANALYTICS_PATH for Train tab

            if os.path.exists(filepath):
                data = load_json_data(filepath)

                if isinstance(data, list):
                    percentages = []
                    query_tokens = []
                    relevant_paragraphs_tokens = []
                    total_paragraphs_tokens = []
                    unique_cases = set()
                    total_paragraphs = []

                    for item in data:
                        if 'file_meta_data_information' in item:
                            for meta_data in item['file_meta_data_information']:
                                percentages.append(meta_data.get('percentage', 0) * 100)
                                query_tokens.append(meta_data.get('query_tokens', 0))
                                relevant_paragraphs_tokens.extend(meta_data.get('relevant_paragraphs_tokens', []))
                                case_link = meta_data.get('case_link')
                                total_paragraphs.append(meta_data.get('total_paragraphs', 0))

                                if case_link not in unique_cases:
                                    total_paragraphs_tokens.extend(meta_data.get('total_paragraphs_tokens', []))
                                    unique_cases.add(case_link)

                    # Calculate the statistics
                    mean_percentage = np.mean(percentages)
                    median_percentage = np.median(percentages)
                    mean_tokens = np.mean(query_tokens)
                    median_tokens = np.median(query_tokens)
                    mean_relevant_tokens = np.mean(relevant_paragraphs_tokens)
                    median_relevant_tokens = np.median(relevant_paragraphs_tokens)
                    mean_total_tokens = np.mean(total_paragraphs_tokens)
                    median_total_tokens = np.median(total_paragraphs_tokens)

                    percentage_counts = Counter(percentages)
                    query_tokens_counts = Counter(query_tokens)
                    relevant_tokens_counts = Counter(relevant_paragraphs_tokens)
                    total_tokens_counts = Counter(total_paragraphs_tokens)

                    # Plot the relevant graphs
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.bar(percentage_counts.keys(), percentage_counts.values(), color='salmon', width=0.2)
                    ax1.axvline(mean_percentage, color='black', linestyle='--', label='Mean')
                    ax1.axvline(median_percentage, color='orange', linestyle='--', label='Median')
                    ax1.set_xlim(-2, max(percentage_counts.keys()) + 5)
                    ax1.set_ylim(0, max(percentage_counts.values()) + 5)
                    ax1.set_title('% of relevant paragraphs per judgement')
                    ax1.set_xlabel('Percentage')
                    ax1.set_ylabel('Frequency')
                    ax1.legend()
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.bar(query_tokens_counts.keys(), query_tokens_counts.values(), color='#D8BFD8', width=1)
                    ax2.axvline(mean_tokens, color='black', linestyle='--', label='Mean')
                    ax2.axvline(median_tokens, color='orange', linestyle='--', label='Median')
                    ax2.set_xlim(-2, max(query_tokens_counts.keys()) + 2)
                    ax2.set_ylim(0, max(query_tokens_counts.values()) + 10)
                    ax2.set_title('Number of tokens per query')
                    ax2.set_xlabel('Query Tokens')
                    ax2.set_ylabel('Frequency')
                    ax2.legend()
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    ax3.bar(relevant_tokens_counts.keys(), relevant_tokens_counts.values(), color='lightblue', width=10)
                    ax3.axvline(mean_relevant_tokens, color='black', linestyle='--', label='Mean')
                    ax3.axvline(median_relevant_tokens, color='orange', linestyle='--', label='Median')
                    ax3.set_xlim(-200, max(relevant_tokens_counts.keys()) + 20)
                    ax3.set_ylim(0, max(relevant_tokens_counts.values()) + 5)
                    ax3.set_title('Relevant Paragraph Tokens Distribution')
                    ax3.set_xlabel('Relevant Paragraph Tokens')
                    ax3.set_ylabel('Frequency')
                    ax3.legend()
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    ax4.bar(total_tokens_counts.keys(), total_tokens_counts.values(), color='lightgreen', width=10)
                    ax4.axvline(mean_total_tokens, color='black', linestyle='--', label='Mean')
                    ax4.axvline(median_total_tokens, color='orange', linestyle='--', label='Median')
                    ax4.set_xlim(-200, max(total_tokens_counts.keys()) + 20)
                    ax4.set_ylim(0, max(total_tokens_counts.values()) + 10)
                    ax4.set_title('Total Paragraph Tokens Distribution (Log Scale on Y-Axis)')
                    ax4.set_xlabel('Total Paragraph Tokens')
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                    st.pyplot(fig4)
                    
                    mean_total_paragraphs = np.mean(total_paragraphs)
                    median_total_paragraphs = np.median(total_paragraphs)
                    total_paragraphs_counts = Counter(total_paragraphs)

                    # New plot for total paragraphs distribution
                    fig6, ax6 = plt.subplots(figsize=(8, 6))
                    ax6.bar(total_paragraphs_counts.keys(), total_paragraphs_counts.values(), color='#8FBC8F', width=1)  # Soft mint green
                    ax6.axvline(mean_total_paragraphs, color='black', linestyle='--', label='Mean')
                    ax6.axvline(median_total_paragraphs, color='orange', linestyle='--', label='Median')
                    ax6.set_xlim(0, max(total_paragraphs_counts.keys()) + 20)
                    ax6.set_ylim(0, max(total_paragraphs_counts.values()) + 5)
                    ax6.set_title('Total Paragraphs Distribution per Case')
                    ax6.set_xlabel('Number of Total Paragraphs')
                    ax6.set_ylabel('Frequency')
                    ax6.legend()
                    st.pyplot(fig6)

                else:
                    st.error("The selected JSON file does not contain a list of data points.")
            else:
                st.error("File not found. Please check the filename and try again.")
                
        st.title("Query-Doc Pairs and Unique Queries Across All Languages")

        languages_data = []
        for lang in LANGUAGES:
            counts_filepath = os.path.join(UNSEEN_QUERIES_VAL_COUNTS_PATH, f"{lang}.json")
            if os.path.exists(counts_filepath):
                with open(counts_filepath, 'r') as f:
                    counts_data = json.load(f)
                num_q_d_pairs = counts_data.get("number_of_q_d_pairs", 0)
                num_unique_queries = counts_data.get("number_of_unique_queries", 0)
                languages_data.append((lang.capitalize(), num_q_d_pairs, num_unique_queries))

        labels = [item[0] for item in languages_data]
        q_d_pairs = [item[1] for item in languages_data]
        unique_queries = [item[2] for item in languages_data]

        x = np.arange(len(labels))
        width = 0.35

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        bars1 = ax6.bar(x - width/2, q_d_pairs, width, label='Query-Doc Pairs')
        bars2 = ax6.bar(x + width/2, unique_queries, width, label='Unique Queries')

        ax6.set_xlabel('Languages')
        ax6.set_ylabel('Count')
        ax6.set_title('Query-Doc Pairs and Unique Queries by Language')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.legend()

        def autolabel(bars):
            """Attach a text label above each bar in *bars*, displaying its height."""
            for bar in bars:
                height = bar.get_height()
                ax6.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)

        st.pyplot(fig6)


    with unique_query_test_tab:
        st.title("Unique Query Test Data Analytics")
        st.write("_Tokens here refer to BERT Tokenizer generated tokens._")
        # Select language
        language_unique_test = st.selectbox("Select language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LANGUAGE), key="language_unique_test")
        
        if language_unique_test:
            filename = f"{language_test}.json"
            filepath = os.path.join(UNSEEN_QUERIES_UNIQUE_QUERY_ANALYTICS_PATH, filename)  # Use TRAIN_ANALYTICS_PATH for Train tab

            if os.path.exists(filepath):
                data = load_json_data(filepath)

                if isinstance(data, list):
                    percentages = []
                    query_tokens = []
                    relevant_paragraphs_tokens = []
                    total_paragraphs_tokens = []
                    unique_cases = set()
                    total_paragraphs = []

                    for item in data:
                        if 'file_meta_data_information' in item:
                            for meta_data in item['file_meta_data_information']:
                                percentages.append(meta_data.get('percentage', 0) * 100)
                                query_tokens.append(meta_data.get('query_tokens', 0))
                                relevant_paragraphs_tokens.extend(meta_data.get('relevant_paragraphs_tokens', []))
                                case_link = meta_data.get('case_link')
                                total_paragraphs.append(meta_data.get('total_paragraphs', 0))

                                if case_link not in unique_cases:
                                    total_paragraphs_tokens.extend(meta_data.get('total_paragraphs_tokens', []))
                                    unique_cases.add(case_link)

                    # Calculate the statistics
                    mean_percentage = np.mean(percentages)
                    median_percentage = np.median(percentages)
                    mean_tokens = np.mean(query_tokens)
                    median_tokens = np.median(query_tokens)
                    mean_relevant_tokens = np.mean(relevant_paragraphs_tokens)
                    median_relevant_tokens = np.median(relevant_paragraphs_tokens)
                    mean_total_tokens = np.mean(total_paragraphs_tokens)
                    median_total_tokens = np.median(total_paragraphs_tokens)

                    percentage_counts = Counter(percentages)
                    query_tokens_counts = Counter(query_tokens)
                    relevant_tokens_counts = Counter(relevant_paragraphs_tokens)
                    total_tokens_counts = Counter(total_paragraphs_tokens)

                    # Plot the relevant graphs
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.bar(percentage_counts.keys(), percentage_counts.values(), color='salmon', width=0.2)
                    ax1.axvline(mean_percentage, color='black', linestyle='--', label='Mean')
                    ax1.axvline(median_percentage, color='orange', linestyle='--', label='Median')
                    ax1.set_xlim(-2, max(percentage_counts.keys()) + 5)
                    ax1.set_ylim(0, max(percentage_counts.values()) + 5)
                    ax1.set_title('% of relevant paragraphs per judgement')
                    ax1.set_xlabel('Percentage')
                    ax1.set_ylabel('Frequency')
                    ax1.legend()
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.bar(query_tokens_counts.keys(), query_tokens_counts.values(), color='#D8BFD8', width=1)
                    ax2.axvline(mean_tokens, color='black', linestyle='--', label='Mean')
                    ax2.axvline(median_tokens, color='orange', linestyle='--', label='Median')
                    ax2.set_xlim(-2, max(query_tokens_counts.keys()) + 2)
                    ax2.set_ylim(0, max(query_tokens_counts.values()) + 10)
                    ax2.set_title('Number of tokens per query')
                    ax2.set_xlabel('Query Tokens')
                    ax2.set_ylabel('Frequency')
                    ax2.legend()
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    ax3.bar(relevant_tokens_counts.keys(), relevant_tokens_counts.values(), color='lightblue', width=10)
                    ax3.axvline(mean_relevant_tokens, color='black', linestyle='--', label='Mean')
                    ax3.axvline(median_relevant_tokens, color='orange', linestyle='--', label='Median')
                    ax3.set_xlim(-200, max(relevant_tokens_counts.keys()) + 20)
                    ax3.set_ylim(0, max(relevant_tokens_counts.values()) + 5)
                    ax3.set_title('Relevant Paragraph Tokens Distribution')
                    ax3.set_xlabel('Relevant Paragraph Tokens')
                    ax3.set_ylabel('Frequency')
                    ax3.legend()
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    ax4.bar(total_tokens_counts.keys(), total_tokens_counts.values(), color='lightgreen', width=10)
                    ax4.axvline(mean_total_tokens, color='black', linestyle='--', label='Mean')
                    ax4.axvline(median_total_tokens, color='orange', linestyle='--', label='Median')
                    ax4.set_xlim(-200, max(total_tokens_counts.keys()) + 20)
                    ax4.set_ylim(0, max(total_tokens_counts.values()) + 10)
                    ax4.set_title('Total Paragraph Tokens Distribution (Log Scale on Y-Axis)')
                    ax4.set_xlabel('Total Paragraph Tokens')
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                    st.pyplot(fig4)

                    
                    mean_total_paragraphs = np.mean(total_paragraphs)
                    median_total_paragraphs = np.median(total_paragraphs)
                    total_paragraphs_counts = Counter(total_paragraphs)

                    # New plot for total paragraphs distribution
                    fig6, ax6 = plt.subplots(figsize=(8, 6))
                    ax6.bar(total_paragraphs_counts.keys(), total_paragraphs_counts.values(), color='#8FBC8F', width=1)  # Soft mint green
                    ax6.axvline(mean_total_paragraphs, color='black', linestyle='--', label='Mean')
                    ax6.axvline(median_total_paragraphs, color='orange', linestyle='--', label='Median')
                    ax6.set_xlim(0, max(total_paragraphs_counts.keys()) + 20)
                    ax6.set_ylim(0, max(total_paragraphs_counts.values()) + 5)
                    ax6.set_title('Total Paragraphs Distribution per Case')
                    ax6.set_xlabel('Number of Total Paragraphs')
                    ax6.set_ylabel('Frequency')
                    ax6.legend()
                    st.pyplot(fig6)
                    
                else:
                    st.error("The selected JSON file does not contain a list of data points.")
            else:
                st.error("File not found. Please check the filename and try again.")
                
        st.title("Query-Doc Pairs and Unique Queries Across All Languages")

        languages_data = []
        for lang in LANGUAGES:
            counts_filepath = os.path.join(UNSEEN_QUERIES_UNIQUE_QUERY_COUNTS_PATH, f"{lang}.json")
            if os.path.exists(counts_filepath):
                with open(counts_filepath, 'r') as f:
                    counts_data = json.load(f)
                num_q_d_pairs = counts_data.get("number_of_q_d_pairs", 0)
                num_unique_queries = counts_data.get("number_of_unique_queries", 0)
                languages_data.append((lang.capitalize(), num_q_d_pairs, num_unique_queries))

        labels = [item[0] for item in languages_data]
        q_d_pairs = [item[1] for item in languages_data]
        unique_queries = [item[2] for item in languages_data]

        x = np.arange(len(labels))
        width = 0.35

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        bars1 = ax6.bar(x - width/2, q_d_pairs, width, label='Query-Doc Pairs')
        bars2 = ax6.bar(x + width/2, unique_queries, width, label='Unique Queries')

        ax6.set_xlabel('Languages')
        ax6.set_ylabel('Count')
        ax6.set_title('Query-Doc Pairs and Unique Queries by Language')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.legend()

        def autolabel(bars):
            """Attach a text label above each bar in *bars*, displaying its height."""
            for bar in bars:
                height = bar.get_height()
                ax6.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)

        st.pyplot(fig6)

