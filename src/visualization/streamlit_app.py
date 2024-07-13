import streamlit as st
import json
import os
from datetime import datetime

# Base path for JSON files
BASE_PATH = "/home/upadro/code/thesis/output"
RESULTS_PATH = "/home/upadro/code/thesis/output/visualizations"
LANGUAGES = ["arabic", "english", "italian", "romanian", "russian", "turkish", "ukrainian"]

# Function to save computation result to a JSON file
def save_computation(data, is_satisfactory, language):
    data['is_satisfactory'] = is_satisfactory
    results_ = []
    results_file_path = os.path.join(RESULTS_PATH, f"{language}_results.json")
    try:
        # Open the JSON file and attempt to load its contents
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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["JSON Checker", "Results Calculation"])

if page == "JSON Checker":
    # Title of the app
    st.title("JSON Satisfactory Checker")

    # Dropdown for selecting language
    language = st.selectbox("Select language", LANGUAGES)

    if language:
        foldername = os.path.join(BASE_PATH, language, "analysis")
        if os.path.exists(foldername):
            json_files = list_json_files(foldername)
            
            if json_files:
                # Select box to choose a JSON file
                filename = st.selectbox("Select a JSON file", json_files)
                filepath = os.path.join(foldername, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as file:
                        json_data = json.load(file)
                    
                    # Ensure the JSON data is a list of objects
                    if isinstance(json_data, list):
                        # Initialize session state for pagination
                        if 'page' not in st.session_state:
                            st.session_state.page = 0

                        # Create two columns
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            # Display the current JSON object
                            current_data = json_data[st.session_state.page]
                            st.write(f"Displaying JSON object {st.session_state.page + 1} of {len(json_data)}:", current_data)

                        with col2:
                            # Checkbox for satisfactory status
                            is_satisfactory = st.checkbox("Is the JSON satisfactory?", key=f"checkbox_{st.session_state.page}")

                            # Button to save computation
                            if st.button("Save Computation"):
                                save_filename = save_computation(current_data, is_satisfactory, language)
                                st.success(f"Computation saved to {save_filename}")

                        # Navigation buttons at the bottom
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.session_state.page > 0:
                                if st.button("Previous"):
                                    st.session_state.page -= 1
                                    st.experimental_rerun()
                        with col3:
                            if st.session_state.page < len(json_data) - 1:
                                if st.button("Next"):
                                    st.session_state.page += 1
                                    st.experimental_rerun()
                    else:
                        st.error("The provided JSON file is not a list of objects.")
                else:
                    st.error("File not found. Please check the filename and try again.")
            else:
                st.error("No JSON files found in the specified folder.")
        else:
            st.error("Folder not found. Please check the folder name and try again.")
elif page == "Results Calculation":
    # Title of the app
    st.title("Results Calculation")

    language = st.selectbox("Select language", LANGUAGES)

    if language:
        foldername = RESULTS_PATH
        results_file_path = os.path.join(foldername, f"{language}_results.json")
        
        if os.path.exists(results_file_path):
            true_count, false_count = calculate_satisfactory_results(results_file_path)
            
            st.write(f"Number of satisfactory results: {true_count}")
            st.write(f"Number of unsatisfactory results: {false_count}")
        else:
            st.error("Results file not found. Please check the filename and try again.")
