import streamlit as st
import geopandas as gpd
import folium
import os
from streamlit_folium import folium_static
import datetime
import pandas as pd
import requests
import urllib.parse
import openai
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from fpdf import FPDF
import pdfplumber
from io import BytesIO
import openpyxl
from openpyxl import Workbook
import xlsxwriter
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from collections import Counter
from google.cloud import translate_v2 as translate
import logging
import math
from urllib.request import urlopen
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import string
import tiktoken
from dateutil.parser import parse
# from datetime import datetime

# logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
cse_id = os.getenv('CSE_ID')
api_key = os.getenv('API_KEY')

data = gpd.read_file(os.path.join('data', 'merged_file.gpkg'))
data = data[data['field_3'].notna()]

# Function to load language codes from a CSV file and create a mapping
def load_language_codes(filename):
    df = pd.read_csv(filename)
    return dict(zip(df['Name'], df['Code']))


# Function to translate text using the Google Cloud Translation API
def translate_text_with_google_cloud(text, language_name):
    # Load the language codes
    language_codes = load_language_codes('data/languages_codes.csv')

    # Get the language code from the language name
    target_language = language_codes.get(language_name)
    target_language = 'en'

    if not target_language:
        raise ValueError(f"Invalid language name: {language_name}")

    url = "https://translation.googleapis.com/language/translate/v2"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    params = {
        'q': text,
        'target': target_language,
        'format': 'text',
        'key': api_key
    }
    response = requests.post(url, params=params)
    try:
        #if response.status_code == 200:
        result = response.json()
        translated_text = result['data']['translations'][0]['translatedText']
        return translated_text
    except:
        error_text = "Error translating: " + text
        return error_text
            
def welcome_page():
    st.title("Welcome to the Energy, Policy, and News Tracker")
    st.markdown("""
    This application assists in tracking energy policies and news across different regions. Here's a guide to each section:

    - **🌍 Area Selection**: Select the geographical area for your research.
    - **✅ Selected Area Check**: Verify the selected areas.
    - **🛠️ Define Research**: Customize your research parameters.
    - **🔍 Research**: Conduct the research based on your parameters.
    - **💻 Pre-processing**: Process the gathered data for analysis.
    - **📚 Document Analysis**: Analyze the processed data.

    Navigate through each section using the sidebar. Begin by selecting the area in '🌍 Area Selection'.
    """)

def area_selection():
    st.markdown("""
    This section allows you to select the geographical area for your research. Choose from various categories such as Country, Continent, WEO Region, or World. Your selection will determine the scope of the data and information that the application will gather and analyze in later steps.
    """)

    # Initialize session state variables if they don't exist
    if 'selected_countries' not in st.session_state:
        st.session_state.selected_countries = []
    if 'subset_data' not in st.session_state:
        st.session_state.subset_data = None
    if 'country_codes' not in st.session_state:
        st.session_state.country_codes = []

    # Sidebar menu for selecting the category
    menu_options = ['Country', 'Continent', 'WEO Region', 'UN Region', 'World', 'No specific area']
    selection = st.sidebar.radio("Choose a category", menu_options, index=0)

    data = gpd.read_file(os.path.join('data', 'merged_file.gpkg'))
    data = data[data['field_3'].notna()]

    if selection == 'Country':
        countries = sorted(data['field_3'].unique().tolist())
        selected_countries = st.sidebar.multiselect('Choose countries', countries, default=st.session_state.selected_countries)
        st.session_state.selected_countries = selected_countries
        subset_data = data[data['field_3'].isin(selected_countries)]

    elif selection == 'Continent':
        continents = sorted(data['continent'].unique().tolist())
        selected_continent = st.sidebar.selectbox('Choose a continent', continents, index=continents.index('Europe'))
        subset_countries = sorted(data[data['continent'] == selected_continent]['field_3'].unique().tolist())
        selected_countries = st.sidebar.multiselect('Choose countries', subset_countries, default=subset_countries)
        subset_data = data[data['field_3'].isin(selected_countries)]
        st.session_state.selected_countries = selected_countries

    elif selection == 'WEO Region':
        weo_regions = [x for x in data['Code_Region'].unique() if x is not None]
        weo_regions = sorted(weo_regions)
        selected_weo = st.sidebar.selectbox('Choose a WEO Region', weo_regions, index=weo_regions.index('EUA'))
        subset_countries = sorted(data[data['Code_Region'] == selected_weo]['field_3'].unique().tolist())
        selected_countries = st.sidebar.multiselect('Choose countries', subset_countries, default=subset_countries)
        subset_data = data[data['field_3'].isin(selected_countries)]
        st.session_state.selected_countries = selected_countries

    elif selection == 'UN Region':
        un_regions = sorted([x for x in data['region'].unique() if x is not None])
        selected_un_region = st.sidebar.selectbox('Choose a UN Region', un_regions)
        subset_countries = sorted(data[data['region'] == selected_un_region]['field_3'].unique().tolist())
        selected_countries = st.sidebar.multiselect('Choose countries', subset_countries, default=subset_countries)
        subset_data = data[data['field_3'].isin(selected_countries)]
        st.session_state.selected_countries = selected_countries

    elif selection == 'World':
        subset_data = data
        st.session_state.selected_countries = list(data.field_3)

    elif selection == 'No specific area':
        st.session_state.subset_data = None
        st.session_state.selected_countries = []
        st.write("You didn't select any area, so the following research will be a general Google search on the topics you'll select.")

    if selection != 'No specific area':
        st.session_state.subset_data = subset_data


    # Read the CSV
    tld_data = pd.read_csv(os.path.join('data', 'tld.csv'), encoding='utf-8')
    # Extracting the TLDs based on selected countries
    selected_tlds = tld_data[tld_data['country'].isin(st.session_state.selected_countries)]['tld'].tolist()
    
    st.session_state.country_codes = selected_tlds

    # You should include this part outside the `elif` statements
    if selection != 'No specific area':
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")
        folium.GeoJson(subset_data).add_to(m)
        folium_static(m)


def selected_area_check():
    if 'subset_data' in st.session_state and st.session_state.subset_data is not None:
        st.write("### Check the table below, and confirm it's the region you are interested in.")
        st.write("If it matches your criteria, proceed to the next step. Otherwise, return to the 'Area Selection' step to adjust your choices.")

        st.table(st.session_state.subset_data[['field_3', 'continent', 'Code_Region']].rename(columns={'field_3': 'Country'}))

        st.markdown("""
        Here, you can review the geographical areas you have selected. This step is crucial to ensure that the regions of interest are correctly chosen before proceeding with the research. If the selected areas don't match your criteria, you can return to the 'Area Selection' step to adjust your choices.
        """)
        
    else:
        st.warning("No countries selected in the 'Area Selection' step.")

def define_research():
    # Ensure that the necessary data is in the session state
    if 'subset_data' not in st.session_state or st.session_state.subset_data is None:
        st.warning("Please complete the previous steps first.")
        return

    st.title("Research Customization")
    st.markdown("""
    Customize your research parameters in this section. You can select the type of research (policies, news, projects), choose information sources, set languages, and define keywords. These parameters will guide the data collection process, ensuring that the research is tailored to your specific needs and interests.
    """)

    # Initialize variables in session state if not present
    if 'config_type_choice' not in st.session_state:
        st.session_state.config_type_choice = 'Customize'  # or 'Predefined' as default

    if 'selected_research_type' not in st.session_state:
        st.session_state.selected_research_type = None  # Default value for research type

    if 'config_choice' not in st.session_state:
        st.session_state.config_choice = None  # Default value for configuration choice
        
    # Function to convert date strings to datetime objects
    def parse_date(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    # Function to convert 'True'/'False' strings to booleans
    def parse_boolean(bool_str):
        return True if bool_str == 'TRUE' else False

   
    if 'subset_data' in st.session_state and st.session_state.subset_data is not None:
       

        defaults = {
            'research_type': 'policies',
            'sources': 'predefined sources search',
            'limit_to_country': True,
            'official_sources': [],
            'selected_predefined_links': [],
            'start_date': datetime.date.today() - datetime.timedelta(days=1),
            'end_date': datetime.date.today(),
            'selected_language': [],
            'selected_mandatory_keywords': [],
            'selected_keywords': [],
            'selected_comp_keywords': [],
            'include_monetary_info': False,
            'selected_countries': [],  # Assuming you need this
            'main_selected_translations': {},
            'comp_selected_translations': {},
            'mandatory_selected_translations': {},
            'translated_trans_keywords': [],
            'final_selected_keywords': []
            
        }

        # Allow user to choose between predefined or custom configuration
        config_type_choice = st.sidebar.radio("Configuration Type", ('Predefined', 'Customize'), index=('Predefined', 'Customize').index(st.session_state.config_type_choice))
        st.session_state.config_type_choice = config_type_choice


        if config_type_choice == "Predefined":
            predefined_configs_df = pd.read_csv('data/predefined_configs.csv')

            # Dropdown to filter by research type
            unique_research_types = predefined_configs_df['Research Type'].unique()
            # Use session state value as default if available
            default_research_type_index = 0 if st.session_state.selected_research_type not in unique_research_types else list(unique_research_types).index(st.session_state.selected_research_type)
            selected_research_type = st.sidebar.radio("Select Research Type", unique_research_types, index=default_research_type_index)
            

            # Filter configurations by selected research type
            filtered_configs = predefined_configs_df[predefined_configs_df['Research Type'] == selected_research_type]

            # Dropdown to select a specific configuration based on the research type
            if st.session_state.config_choice not in filtered_configs['Config Name'].tolist():
                st.session_state.config_choice = None  # Reset if previous choice is not in the new list
            config_choice = st.sidebar.selectbox("Choose Configuration", filtered_configs['Config Name'].tolist(), index=filtered_configs['Config Name'].tolist().index(st.session_state.config_choice) if st.session_state.config_choice in filtered_configs['Config Name'].tolist() else 0)
            
        
            if config_choice:
                # Display the details of the chosen configuration
                selected_config = filtered_configs[filtered_configs['Config Name'] == config_choice].iloc[0]
                st.dataframe(selected_config.to_frame())

                st.session_state.research_type = selected_config['Research Type']
                
                # Load necessary data based on the selected research type
                if st.session_state.research_type == "spending tracker":
                    links_df = pd.read_csv('data/links.csv', encoding='utf-8')
                elif st.session_state.research_type == "news":
                    links_df = pd.read_csv('data/news_links.csv', encoding='utf-8')
                else:  # For "policies" and "projects"
                    links_df = pd.read_csv('data/energy_stakeholders_links.csv', encoding='utf-8')

                st.session_state.sources = selected_config['Sources']
                st.session_state.limit_to_country = selected_config['Limit to Country'] == 'TRUE'
                st.session_state.official_sources = selected_config['Official Sources'].split(';')
                st.session_state.selected_mandatory_keywords = selected_config['Mandatory Keywords'].split(';')
                st.session_state.selected_keywords = selected_config['Topic Keywords'].split(';')
                st.session_state.selected_comp_keywords = selected_config['Complementary Keywords'].split(';')
                
                languages_df = pd.read_csv('data/languages.csv', encoding='utf-8') 
                comp_keywords_df = pd.read_csv('data/complementary_keywords.csv', encoding='utf-8')
                mandatory_keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
                keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
            
                all_languages = languages_df.melt(id_vars=['Country'], 
                                value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']
                                ).dropna()['value'].unique().tolist()

                # Determine the default languages based on the selected countries
                if len(st.session_state.selected_countries) > 1:
                    default_languages = ["English"]
                elif not st.session_state.subset_data.empty:
                    selected_country_languages = languages_df[languages_df['Country'].isin(st.session_state.selected_countries)]
                    default_languages = selected_country_languages.melt(id_vars=['Country'], 
                                                                        value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']
                                                                    ).dropna()['value'].unique().tolist()
                else:
                    default_languages = ["English"]

                # Language selection multiselect widget
                st.session_state.selected_language = st.multiselect("Language(s):",
                                                                    sorted(all_languages),
                                                                    default=default_languages,
                                                                    help="Choose the languages for your research. This will filter content based on the selected languages.")


                st.session_state.selected_research_type = selected_research_type
                st.session_state.config_choice = config_choice



        if config_type_choice == "Customize":    
            for key, value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = value

            # 1. Kind of Research
            st.subheader("1. Research Type")

            # Check if 'selected_research_type' is already set in session_state
            if 'selected_research_type' not in st.session_state:
                # Default index if not set
                st.session_state.selected_research_type = 2

            # Set the radio button with the index from session_state
            selected_index = st.session_state.selected_research_type
            research_options = ["policies", "news", "spending tracker", "projects"]
            st.session_state.research_type = st.radio(
                "",
                research_options,
                index=selected_index,
                help="Select the type of research you're interested in. Options include policies, news, and projects."
            )

# Update the session_state with the current selection
st.session_state.selected_research_type = research_options.index(st.session_state.research_type)

            # Load necessary data based on the selected research type
            if st.session_state.research_type == "spending tracker":
                links_df = pd.read_csv('data/links.csv', encoding='utf-8')
            elif st.session_state.research_type == "news":
                links_df = pd.read_csv('data/news_links.csv', encoding='utf-8')
            else:  # For "policies" and "projects"
                links_df = pd.read_csv('data/energy_stakeholders_links.csv', encoding='utf-8')

            # Separator
            st.markdown("---")
        
            languages_df = pd.read_csv('data/languages.csv', encoding='utf-8')
            comp_keywords_df = pd.read_csv('data/complementary_keywords.csv', encoding='utf-8')
            mandatory_keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
            keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')

            # 0. Administrative division
            if 'selected_countries' in st.session_state and not st.session_state.subset_data.empty:
                # Check if only one country is selected
                if len(st.session_state.selected_countries) == 1 and st.session_state.research_type == "spendings":
                    selected_country = st.session_state.selected_countries[0]

                    # Filter the DataFrame for the selected country
                    country_df = links_df[links_df['Country'] == selected_country]

                    # Extract unique regions from the DataFrame
                    unique_regions = country_df['Region'].unique()

                    # Create a dropdown menu for regions
                    selected_region = st.selectbox("Select a Region", unique_regions)

                    # Update links_df to filter by the selected region
                    links_df = country_df[country_df['Region'] == selected_region]


            # 2. Sources of Information
            st.subheader("2. Information Sources")
            if not st.session_state.subset_data.empty:
                st.session_state.sources = st.radio("Choose a source:",
                                                ["predefined sources search", "general google search", "general twitter search", "general linkedin search"],
                                                index=["predefined sources search", "general google search", "general twitter search", "general linkedin search"].index(st.session_state.sources),
                                                help="Select your preferred sources of information.")

                # Limit research to country-specific domains
                if st.session_state.sources == "general google search":
                    st.session_state.limit_to_country = st.checkbox("Limit research to country-specific domains?", value=st.session_state.limit_to_country, help="This might provide precise results for country-specific information but will exclude international sources with .com, .org, etc.")

                # 2.1 Official Sources (if selected)
                if st.session_state.sources == "predefined sources search":
                    types_list = links_df.loc[links_df['Country'].isin(st.session_state.selected_countries), 'Type'].unique().tolist()
                    st.session_state.official_sources = st.multiselect("",
                                                                    types_list,
                                                                    default=types_list,
                                                                    help="Select official sources for predefined sources search.")

                    source_counts = links_df[links_df['Country'].isin(st.session_state.selected_countries)].groupby(['Type', 'Country']).size().unstack(fill_value=0)
                    st.write(source_counts)

                    links_list = list(set(links_df[(links_df['Country'].isin(st.session_state.selected_countries)) & (links_df['Type'].isin(st.session_state.official_sources))].Link))
                    
                    st.session_state.selected_predefined_links = [x for x in links_list if not (isinstance(x, float) and math.isnan(x))]

            else:
                st.session_state.sources = st.radio("Choose a source:",
                                                ["general google search", "general twitter search", "general linkedin search"],
                                                index=["general google search", "general twitter search", "general linkedin search"].index(st.session_state.sources),
                                                help="Select your preferred sources of information.")

            # Separator
            st.markdown("---")

            # 3. Language
            st.subheader("3. Language")

            # Extract all unique languages from the dataframe
            all_languages = languages_df.melt(id_vars=['Country'], 
                                            value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']
                                            ).dropna()['value'].unique().tolist()

            # Determine the default languages based on the selected countries
            if len(st.session_state.selected_countries) > 1:
                default_languages = ["English"]
            elif not st.session_state.subset_data.empty:
                selected_country_languages = languages_df[languages_df['Country'].isin(st.session_state.selected_countries)]
                default_languages = selected_country_languages.melt(id_vars=['Country'], 
                                                                    value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']
                                                                ).dropna()['value'].unique().tolist()
            else:
                default_languages = ["English"]

            # Language selection multiselect widget
            st.session_state.selected_language = st.multiselect("Language(s):",
                                                                sorted(all_languages),
                                                                default=default_languages,
                                                                help="Choose the languages for your research. This will filter content based on the selected languages.")


            # Separator
            st.markdown("---")

            # 4. Mandatory Keywords
            st.subheader("4. Mandatory Keywords")
            mandatory_keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
            all_mandatory_keywords = sorted(set(mandatory_keywords_df['keyword'].tolist()))
            
            if 'selected_mandatory_keywords' not in st.session_state:
                st.session_state.selected_mandatory_keywords = []

            if 'selected_mandatory_keywords' in st.session_state:
                all_mandatory_keywords = sorted(list(set(list(all_mandatory_keywords) + list(st.session_state.selected_mandatory_keywords))))
                
            additional_mandatory_keywords = st.text_area("Additional Mandatory Keywords (comma-separated):")

            if additional_mandatory_keywords:
                additional_keywords_list = [kw.strip() for kw in additional_mandatory_keywords.split(",")]
                for kw in additional_keywords_list:
                    if kw not in all_mandatory_keywords:
                        all_mandatory_keywords.append(kw)
                    if kw not in st.session_state.selected_mandatory_keywords:
                        st.session_state.selected_mandatory_keywords.append(kw)

            st.session_state.selected_mandatory_keywords = st.multiselect("Mandatory Keywords:",
                                                                    all_mandatory_keywords,
                                                                    default=st.session_state.selected_mandatory_keywords,
                                                                    help="Select mandatory keywords. These are essential terms that MUST appear in the research results.")



            st.markdown("---")
            
            # 5. Topic Keywords
            st.subheader("5. Topic Keywords")
            keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
            all_topic_keywords = sorted(set(keywords_df['keyword'].tolist()))

            if 'selected_keywords' not in st.session_state:
                st.session_state.selected_keywords = []

            if 'selected_keywords' in st.session_state:
                all_topic_keywords = sorted(list(set(list(all_topic_keywords) + list(st.session_state.selected_keywords))))
                
            custom_keywords = st.text_input("Add additional topic keywords (comma separated):")
            if custom_keywords:
                custom_keywords_list = [keyword.strip() for keyword in custom_keywords.split(',')]
                for kw in custom_keywords_list:
                    if kw not in all_topic_keywords:
                        all_topic_keywords.append(kw)
                    if kw not in st.session_state.selected_keywords:
                        st.session_state.selected_keywords.append(kw)

            filtered_topic_keywords = [k for k in all_topic_keywords if k not in st.session_state.selected_mandatory_keywords]
            st.session_state.selected_keywords = st.multiselect("Keywords:",
                                                                filtered_topic_keywords,
                                                                default=st.session_state.selected_keywords,
                                                                help="Choose your topic keywords. These are the main terms related to your research topic. AT LEAST ONE OF THEM NEED TO APPEEAR IN THE DOCUMENT")

            st.markdown("---")

            # 6. Complementary Research Keywords
            st.subheader("6. Complementary Research Keywords")
            comp_keywords_df = pd.read_csv('data/complementary_keywords.csv', encoding='utf-8')
            all_comp_keywords = sorted(set(comp_keywords_df['keyword'].tolist()))

            if 'selected_comp_keywords' not in st.session_state:
                st.session_state.selected_comp_keywords = []

            if 'selected_comp_keywords' in st.session_state:
                all_comp_keywords = sorted(list(set(list(all_comp_keywords) + list(st.session_state.selected_comp_keywords))))
                
            custom_comp_keywords = st.text_input("Add additional complementary keywords (comma separated):")
            if custom_comp_keywords:
                custom_comp_keywords_list = [keyword.strip() for keyword in custom_comp_keywords.split(',')]
                for kw in custom_comp_keywords_list:
                    if kw not in all_comp_keywords:
                        all_comp_keywords.append(kw)
                    if kw not in st.session_state.selected_comp_keywords:
                        st.session_state.selected_comp_keywords.append(kw)

            st.session_state.selected_comp_keywords = st.multiselect("Keywords:",
                                                                    all_comp_keywords,
                                                                    default=st.session_state.selected_comp_keywords,
                                                                    help="Select complementary keywords. These are additional terms that can enhance your research scope. AT LEAST ONE OF THEM NEED TO APPEEAR IN THE DOCUMENT")



        def translate_word(text, language_name):
            # Load the language codes
            language_codes = load_language_codes('data/languages_codes.csv')

            # Get the language code from the language name
            target_language = language_codes.get(language_name)

            if not target_language:
                raise ValueError(f"Invalid language name: {language_name}")

            url = "https://translation.googleapis.com/language/translate/v2"
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            params = {
                'q': text,
                'target': target_language,
                'format': 'text',
                'key': api_key
            }
            response = requests.post(url, params=params)
            try:
                #if response.status_code == 200:
                result = response.json()
                translated_text = result['data']['translations'][0]['translatedText']
                return translated_text
            except:
                error_text = "Error translating: " + text
                return error_text

        # Extract respective translations for the selected keywords
        main_selected_translations = {}
        comp_selected_translations = {}
        mandatory_selected_translations = {}

        def is_quoted(keyword):
            return keyword[1:-1] if keyword.startswith('"') and keyword.endswith('"') else keyword

        for language in st.session_state.selected_language:
            # Translations for mandatory keywords
            mandatory_selected_translations[language] = [
                is_quoted(keyword) if is_quoted(keyword) != keyword else
                (mandatory_keywords_df.loc[mandatory_keywords_df['keyword'] == keyword, language].tolist()[0]
                if keyword in mandatory_keywords_df['keyword'].tolist() else
                translate_word(keyword, st.session_state.selected_language[0]))
                for keyword in st.session_state.selected_mandatory_keywords
            ]

            # Translations for main keywords
            main_selected_translations[language] = [
                is_quoted(keyword) if is_quoted(keyword) != keyword else
                (keywords_df.loc[keywords_df['keyword'] == keyword, language].tolist()[0]
                if keyword in keywords_df['keyword'].tolist() else
                translate_word(keyword, st.session_state.selected_language[0]))
                for keyword in st.session_state.selected_keywords
            ]

            # Translations for complementary keywords
            comp_selected_translations[language] = [
                is_quoted(keyword) if is_quoted(keyword) != keyword else
                (comp_keywords_df.loc[comp_keywords_df['keyword'] == keyword, language].tolist()[0]
                if keyword in comp_keywords_df['keyword'].tolist() else
                translate_word(keyword, st.session_state.selected_language[0]))
                for keyword in st.session_state.selected_comp_keywords
            ]

        # Store translations in session state
        st.session_state.main_selected_translations = main_selected_translations
        st.session_state.comp_selected_translations = comp_selected_translations
        st.session_state.mandatory_selected_translations = mandatory_selected_translations

        # Create a flat list of all translated keywords
        translated_keywords = []

        # Include mandatory translated keywords
        for language, translations in st.session_state.mandatory_selected_translations.items():
            translated_keywords.extend(translations)

        # Include main translated keywords
        for language, translations in st.session_state.main_selected_translations.items():
            translated_keywords.extend(translations)

        # Include complementary translated keywords
        for language, translations in st.session_state.comp_selected_translations.items():
            translated_keywords.extend(translations)
        
        # Removing potential duplicates from translated keywords
        translated_trans_keywords = list(set(translated_keywords))
        
        # Create a flat list of all selected keywords (main and complementary)
        final_selected_keywords = st.session_state.selected_mandatory_keywords + st.session_state.selected_keywords + st.session_state.selected_comp_keywords
        
        # Removing potential duplicates from selected keywords
        final_selected_keywords = list(set(final_selected_keywords))

        # Add to session state
        st.session_state.translated_trans_keywords = translated_trans_keywords
        st.session_state.final_selected_keywords = final_selected_keywords

        # Display the final list of translated and selected keywords
        st.write(translated_trans_keywords)

        # Separator
        st.markdown("---")


def research():

    # Ensure that the necessary data is in the session state
    if 'final_selected_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return

    else:    
        st.title("Research 📚")
        st.markdown("""
        Conduct your research based on the parameters set in the previous steps. This section utilizes custom search engines to gather data and information from various sources. You can review the total results, check individual entries, and gather insights relevant to your selected areas and research criteria.
        """)

    # Sidebar for selecting time period
    period_options = ["custom", "last 24h", "last week", "last two weeks", "last month", 
                      "last three months", "last 6 months", "last year", "last 2y", 
                      "last 3y", "last 4y", "last 5y", "last 10y"]
    st.session_state.selected_period = st.sidebar.selectbox("Select Time Period", period_options)

    # Date input based on the selected time period
    if st.session_state.selected_period == "custom":
        st.session_state.start_date = st.sidebar.date_input("Start Date", value=st.session_state.start_date)
        st.session_state.end_date = st.sidebar.date_input("End Date", value=st.session_state.end_date)
    else:
        st.session_state.end_date = datetime.date.today()
        # Define time deltas for each predefined period
        time_deltas = {
            "last 24h": datetime.timedelta(days=1),
            "last week": datetime.timedelta(weeks=1),
            "last two weeks": datetime.timedelta(weeks=2),
            "last month": datetime.timedelta(weeks=4),
            "last three months": datetime.timedelta(weeks=12),
            "last 6 months": datetime.timedelta(weeks=26),
            "last year": datetime.timedelta(weeks=52),
            "last 2y": datetime.timedelta(weeks=104),
            "last 3y": datetime.timedelta(weeks=156),
            "last 4y": datetime.timedelta(weeks=208),
            "last 5y": datetime.timedelta(weeks=260),
            "last 10y": datetime.timedelta(days=3650)
        }
        st.session_state.start_date = st.session_state.end_date - time_deltas[st.session_state.selected_period]

    # Function to check if the date in the snippet is within the selected time window
    def is_date_within_window(date_text, start_date, end_date):
        try:
            # Attempt to parse the date from the snippet
            snippet_date = parse(date_text, fuzzy=True)
            # Check if the date falls within the start and end dates
            return start_date <= snippet_date.date() <= end_date
        except ValueError:
            # If parsing fails, we can't determine the date, so return False
            return False

    def construct_query():
        query_parts = []
        total_query_elements = 0

        # Function to add query part and count elements
        def add_query_part(query_part):
            nonlocal total_query_elements
            total_query_elements += len(query_part.split(' '))
            query_parts.append(query_part)

        # Incorporate translations into the main keywords query
        main_translations_query_parts = []
        for language, translations in st.session_state.main_selected_translations.items():
            translations_query = " OR ".join([f'"{translation}"' for translation in translations])
            if translations_query:
                main_translations_query_parts.append(translations_query)
        if main_translations_query_parts:
            add_query_part(f"({' OR '.join(main_translations_query_parts)})")
        
        # Mandatory keywords
        mandatory_translations_query_parts = []
        for language, translations in st.session_state.mandatory_selected_translations.items():
            mandatory_translations_query = " AND ".join([f'"{translation}"' for translation in translations])
            if mandatory_translations_query:
                mandatory_translations_query_parts.append(mandatory_translations_query)
        if mandatory_translations_query_parts:
            add_query_part(f"({' AND '.join(mandatory_translations_query_parts)})")

        # Complementary keywords
        comp_translations_query_parts = []
        for language, translations in st.session_state.comp_selected_translations.items():
            comp_translations_query = " OR ".join([f'"{translation}"' for translation in translations])
            if comp_translations_query:
                comp_translations_query_parts.append(comp_translations_query)
        if comp_translations_query_parts:
            add_query_part(f"({' OR '.join(comp_translations_query_parts)})")

        # Date range
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        date_query = f'after:{start_date_str} before:{end_date_str}'
        add_query_part(date_query)

        # Other query elements based on research source
        if st.session_state.sources == "general twitter search":
            add_query_part('site:twitter.com')
        elif st.session_state.sources == "general linkedin search":
            add_query_part('site:linkedin.com')
        elif st.session_state.sources == "general google search":
            excluded_sites_query = " -site:iea.org -site:iea.blob.core.windows.net -site:en.wikipedia.org -site:irena.org"
            add_query_part(excluded_sites_query)
            if st.session_state.limit_to_country and 'selected_countries' in st.session_state:
                country_specific_sites_query = " OR ".join([f"site:{code.lower()}" for code in st.session_state.country_codes])
                add_query_part(f"({country_specific_sites_query})")

        return ' AND '.join(query_parts), total_query_elements

    # Create a sidebar with a checkbox
    exact_keywords = st.sidebar.checkbox('Do you want exact keywords?', value=True)
    query, total_query_elements = construct_query()

    def construct_query_2(base_query, research_type):
        if research_type == "Only PDF":
            # Add ".pdf" at the beginning of the query
            return f"'.pdf' {base_query}"
        elif research_type == "Exclude PDF":
            # Add "-filetype:pdf" at the end of the query
            return f"{base_query} -filetype:pdf"
        else:
            # Mixed research, return the base query
            return base_query

    # Dropdown menu for research type
    research_type = st.sidebar.selectbox(
        "Choose the type of research",
        ["Mixed Research", "Only PDF", "Exclude PDF"],
        index=0  # Default to Mixed Research
    )

    # Unpacking the tuple here
    query = construct_query_2(query, research_type)

    # Check if the checkbox is checked
    if not exact_keywords and query:
        # Remove all apostrophes from the query

        query = query.replace('"', '')


    if st.session_state.sources == "general google search":
        if len(st.session_state.selected_countries) > 1:
            query = " OR ".join([f'"{country}"' for country in st.session_state.selected_countries]) + " " + query 

        else:
            query = str(st.session_state.selected_countries[0]) + " " + query 

    # Function to translate text using the Google Cloud Translation API
    def translate_text_with_google_cloud(text, language_name):
        # Load the language codes
        language_codes = load_language_codes('data/languages_codes.csv')

        # Get the language code from the language name
        target_language = language_codes.get(language_name)
        target_language = 'en'

        if not target_language:
            raise ValueError(f"Invalid language name: {language_name}")

        url = "https://translation.googleapis.com/language/translate/v2"
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        params = {
            'q': text,
            'target': target_language,
            'format': 'text',
            'key': api_key
        }
        response = requests.post(url, params=params)
        try:
            #if response.status_code == 200:
            result = response.json()
            translated_text = result['data']['translations'][0]['translatedText']
            return translated_text
        except:
            error_text = "Error translating: " + text
            return error_text

                
    # Checkbox for summary
    want_summary = st.sidebar.checkbox('Do you want a summary?', value=False)

    # Number of links to include in the summarized analysis
    if want_summary:
        num_links_to_summarize = st.sidebar.slider('Select number of links for summarized analysis', 1, 100, 10)

        # Function to load model and tokenizer for summarization
        def load_model():
            model_name = "t5-small"  # You can choose other models like 't5-base', 't5-large'
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            return model, tokenizer

        model, tokenizer = load_model()

        # Ensure you have the necessary NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')

        def summarize_content(text, max_word_count=1000):
            nltk.data.path.append('/app/nltk_data/')  # Update this path if necessary
            
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)

            # Tokenize and filter stopwords and punctuation from the words in the text
            stop_words = set(stopwords.words('english') + list(string.punctuation))
            words = word_tokenize(text.lower())
            filtered_words = [word for word in words if word not in stop_words]

            # Score sentences based on frequency of words
            word_frequencies = defaultdict(int)
            for word in filtered_words:
                word_frequencies[word] += 1

            sentence_scores = defaultdict(int)
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word in word_frequencies:
                        sentence_scores[sentence] += word_frequencies[word]

            # Sort sentences and construct the summary based on word count
            sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
            summary = []
            word_count = 0
            for sentence in sorted_sentences:
                sentence_word_count = len(word_tokenize(sentence))
                if word_count + sentence_word_count <= max_word_count:
                    summary.append(sentence)
                    word_count += sentence_word_count
                else:
                    break

            return '\n'.join(summary)

        # Function to sort sentences based on the count of different keywords
        def sort_sentences(sentences, keywords):
            def count_unique_keywords(sentence):
                word_counts = Counter(word.lower() for word in sentence.split() if word.lower() in keywords)
                return len(word_counts)

            sorted_sentences = sorted(sentences, key=count_unique_keywords, reverse=True)
            return sorted_sentences
        
        def extract_metadata_and_index(pdf, language = st.session_state.selected_language[0]):
            title_name = 'title'
            if language != 'en':
                title_name = translate_text_with_google_cloud(title_name, language)

            title = pdf.metadata.get(title_name, 'No Title Found')
            index_content = ''

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    table_words = ['content', 'index', 'table']

                    if language == 'en':
                        table_words = [translate_text_with_google_cloud(x, language) for x in table_words]
                    
                    if table_words[0] in text.lower() or table_words[1] in text.lower() or table_words[2] in text.lower():
                        index_content += text + '\n\n'
                        # Optional: break if you only want the first occurrence of 'Contents' or 'Index'
                        break

            return title, index_content.strip()
        
        def extract_sentences_from_pdf(url, keywords):
            response = requests.get(url)
            sentences_with_keywords = []
            num_pages = 0

            if response.status_code == 200:
                with pdfplumber.open(BytesIO(response.content)) as pdf:
                    num_pages = len(pdf.pages)
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            sentences = text.split('.')
                            for sentence in sentences:
                                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                                    sentences_with_keywords.append(sentence.strip())

            return sentences_with_keywords, num_pages

        def highlight_keywords(text, keywords):
            # Create a regex pattern to find the keywords
            pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
            
            # Replace the keywords with a span element that applies the green color style
            highlighted_text = re.sub(pattern, r'<span style="color:green">\1</span>', text, flags=re.IGNORECASE)
            
            return highlighted_text

    # Checkbox for translation
    want_translation = st.sidebar.checkbox('Do you want to translate to English?', value=False)

    if st.sidebar.button("Run Research"):
        # query, total_query_elements = construct_query()
        total_query_elements = total_query_elements + len(st.session_state.selected_predefined_links)

        max_parameters_per_query = 32
        
        # Function to break down the links into chunks of size n
        def chunk_list(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # Breaking down the selected predefined links into chunks if necessary
        if st.session_state.sources == "predefined sources search" and total_query_elements > max_parameters_per_query:
            remaining_elements = max_parameters_per_query - (total_query_elements - len(st.session_state.selected_predefined_links))
            link_chunks = list(chunk_list(st.session_state.selected_predefined_links, remaining_elements))
        else:
            link_chunks = [st.session_state.selected_predefined_links]

        total_results = 0
        google_search_urls = []
        # Initialize a list to hold all results
        results = []

        for chunk_index, links_chunk in enumerate(link_chunks):
            predefined_sites_query = " OR ".join([f"site:{site}" for site in links_chunk])
            # Adjust the main query with the current chunk's predefined sites query
            if 'predefined_sites_query' in query:
                chunk_query = query.replace('predefined_sites_query', predefined_sites_query)
            else:
                chunk_query = query + ' AND (' + predefined_sites_query + ')'

            # URL encode the chunk query string
            encoded_query = urllib.parse.quote_plus(chunk_query)

            # Create the hyperlink URL for the chunk query on Google
            google_search_url = f"https://www.google.com/search?q={encoded_query}"
            google_search_urls.append(google_search_url)

            # Making the API calls
            url = "https://www.googleapis.com/customsearch/v1"
            chunk_results = []
            for start_index in range(1, 101, 10):  # Start indices for pagination
                params = {
                    'q': chunk_query,
                    'cx': cse_id,
                    'key': api_key,
                    'num': 10,
                    'start': start_index
                }
                response = requests.get(url, params=params)
                chunk_results.extend(response.json().get("items", []))

            total_results += len(chunk_results)

            # Append the results of the current chunk to the overall results
            results.extend(chunk_results)

        total_results_summary = f"Total combined estimated results: {total_results} | Sources: "
        source_links = ', '.join([f"[{i + 1}]({url})" for i, url in enumerate(google_search_urls)])
        st.markdown(total_results_summary + source_links)


        # Display the results and process for summary
        progress_bar = st.progress(0)
        for i, result in enumerate(results):
            if want_summary:
                if i >= num_links_to_summarize and want_summary:
                    break

                # Update progress bar
                denominator = min(num_links_to_summarize, len(results))
                progress_bar.progress((i + 1) / denominator)

            else:
                progress_bar.progress((i + 1) / len(results))

            # Extract the date from the snippet
            date_text = result['snippet'].split(' ... ')[0]

            if not is_date_within_window(date_text, st.session_state.start_date, st.session_state.end_date):
                continue

            else:
                snippet_without_date = result['snippet'].replace(date_text, '').strip()

                # Determine the document type
                if '.' in result['link'][-6:]:  # Check if the last part of the URL contains a dot (.)
                    doc_type = result['link'].split('.')[-1]  # Get the file extension
                else:
                    doc_type = "webpage"

                # Check if translation is needed
                if want_translation:
                    # Translate title, snippet, and summary
                    translated_title = translate_text_with_google_cloud(result['title'], st.session_state.selected_language[0])
                    translated_snippet = translate_text_with_google_cloud(snippet_without_date, st.session_state.selected_language[0])

                    # Display translated title and snippet
                    st.subheader(f"[{translated_title}]({result['link']})")
                    st.write(f"Source: {result['displayLink']} | Date: {date_text} | Type: {doc_type}")
                    st.write(f"Snippet: {translated_snippet}")

                else:
                    # Existing code to display the result
                    st.subheader(f"[{result['title']}]({result['link']})")
                    st.write(f"Source: {result['displayLink']} | Date: {date_text} | Type: {doc_type}")
                    st.write(f"Snippet: {snippet_without_date}")


                if want_summary:
                    # Determine if it's a webpage or a PDF and process accordingly
                    if doc_type != "pdf":
                        # Scrape the webpage content
                        response = requests.get(result['link'])
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Remove all script and style elements
                        for script_or_style in soup(["script", "style"]):
                            script_or_style.extract()  # Remove the element

                        # Get text and clean it
                        lines = (line.strip() for line in soup.get_text().splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # Split on double space for multi-headlines
                        text = '\n'.join(chunk for chunk in chunks if chunk)

                        # Summarize the webpage content
                        # st.write(text)
                        summary = summarize_content(text, max_word_count=500)  # Adjust max_length as needed

                        # Translate the summary if needed
                        if want_translation:
                            summary = translate_text_with_google_cloud(summary, st.session_state.selected_language[0])


                        # Split the summary into sentences and join with newline characters
                        summary = sent_tokenize(summary)
                        summary = '\n'.join(summary)
                        
                        # Display the summary in an expander
                        with st.expander("Show Summary"):
                            # summary = highlight_keywords(summary, st.session_state.translated_trans_keywords)
                            st.write(summary)

                    else:
                        with st.expander(f"PDF Document Details: {result['title']}"):
                            response = requests.get(result['link'])
                            if response.status_code == 200:
                                with pdfplumber.open(BytesIO(response.content)) as pdf:
                                    # Extract PDF metadata and index
                                    title, index_content = extract_metadata_and_index(pdf)

                                    # Display title and index/content
                                    st.subheader(f"PDF Title: {title}")
                                    st.write(f"Number of Pages in PDF: {len(pdf.pages)}")
                                    if index_content:
                                        # If index or contents are found, display them
                                        st.write("Index / Contents:")

                                        if want_translation:
                                            st.text(translate_text_with_google_cloud(index_content, st.session_state.selected_language[0]))

                                        else:
                                            st.text(index_content)
                                    else:
                                        # If no index/content, extract and display sentences with keywords
                                        keywords = st.session_state.final_selected_keywords  # Adjust based on your app's structure
                                        extracted_sentences = extract_sentences_from_pdf(result['link'], keywords)[0]
                                        sorted_sentences = sort_sentences(extracted_sentences, keywords)[0:20]  # Adjust number as needed

                                        # Translate extracted sentences if needed and prepare them for display
                                        if want_translation:
                                            translated_sorted_sentences = [translate_text_with_google_cloud(sentence, st.session_state.selected_language[0]) for sentence in sorted_sentences]
                                        else:
                                            translated_sorted_sentences = sorted_sentences

                                        # Display the sentences in an expander
                                        for sentence in translated_sorted_sentences:
                                            st.write(sentence)  # Each sentence will be displayed on a new line
            
                            else:
                                st.error("Failed to access the PDF.")

                st.markdown("---")


        # Create a list of dictionaries from results
        data_list = [{'title': result['title'], 'link': result['link'], 'snippet': result['snippet']} for result in results]
        
        # Convert the list to a DataFrame
        results_df = pd.DataFrame(data_list)

        # Save the DataFrame into a CSV file with the desired filename format
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"results/results_df_{timestamp}_{st.session_state.selected_countries[0]}.csv"
        results_df.to_csv(filename, encoding='utf-8', index=False)
        st.session_state.filename = filename


def run_preprocessing():
    st.title("Run Preprocessing 💻")
    st.markdown("""
    Preprocess the gathered data for analysis in this step. This involves data cleaning, normalization, and preparation for detailed analysis. The process ensures that the data is in the right format and structure, enabling effective and accurate analysis in the next steps.
    """)
   
    # Initialize row_selection in session state if not present
    #if 'row_selection' not in st.session_state:
    #    st.session_state.row_selection = []

    # Ensure that the necessary data is in the session state
    if 'final_selected_keywords' not in st.session_state or 'translated_trans_keywords' not in st.session_state or 'filename' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return

    filename = st.session_state.filename

    # Sidebar checkbox for translation
    translate = st.sidebar.checkbox("Translate to English?", value=False)

    # Load the CSV
    df = pd.read_csv(st.session_state.filename, encoding='utf-8')

    # Create an empty dataframe for sentence-level results
    sentence_df = pd.DataFrame(columns=['title', 'link', 'sentence_id', 'sentence'] + st.session_state.final_selected_keywords)

    # Check if data is already processed
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # Dropdown for sorting
    sort_option = st.sidebar.selectbox('Sort by', ['Overall'] + st.session_state.final_selected_keywords)
    st.session_state.first = False

    # Apply sorting based on the selected option
    def apply_sorting(dataframe, sort_by):
        return dataframe.sort_values(by=sort_by, ascending=False)

    if st.sidebar.button("Run Preprocessing"):
        st.session_state.first = True
        progress_bar = st.sidebar.progress(0)
        total_links = len(df)
        for keyword in st.session_state.final_selected_keywords:
            df[keyword] = 0
        df['word_count'] = 0
        df['Normalized_Count'] = 0


        # Iterate through each link in the dataframe
        for index, row in df.iterrows():
            try:
                response = requests.get(row['link'])
                response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
                
                # Process different file types
                if row['link'].endswith('.pdf'):
                    # Process PDF files
                    with pdfplumber.open(BytesIO(response.content)) as pdf:
                        text_content = ''
                        for page in pdf.pages:
                            text_content += page.extract_text()
                else:
                    # Process HTML files
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text_content = soup.get_text().lower()

                text_content = text_content.lower()

                # Calculate word count
                word_count = len(text_content.split())
                df.at[index, 'word_count'] = word_count

                # Document-level keyword counting based on translated keywords
                for keyword, trans_keyword in zip(st.session_state.final_selected_keywords, st.session_state.translated_trans_keywords):
                    df.at[index, keyword] = text_content.count(trans_keyword.lower())

                # Sentence-level keyword counting based on translated keywords
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text_content)

                # Initialize an empty list to collect sentence data
                sentence_data_list = []

                for sentence_id, sentence in enumerate(sentences, 1):
                    sentence_data = {
                        'title': row['title'],
                        'link': row['link'],
                        'sentence_id': f"{index + 1}_{sentence_id}",
                        'sentence': sentence
                    }

                    # Count keywords in each sentence
                    for keyword, trans_keyword in zip(st.session_state.final_selected_keywords, st.session_state.translated_trans_keywords):
                        pattern = re.compile(rf'\b{re.escape(trans_keyword)}s?\b', re.IGNORECASE)
                        sentence_data[keyword] = len(re.findall(pattern, sentence))

                    sentence_data_list.append(sentence_data)

                # Append the collected sentence data to sentence_df

                new_df = pd.DataFrame(sentence_data_list)
                if not new_df.empty:
                    # Normalization of keyword counts in sentences
                    for keyword in st.session_state.final_selected_keywords:
                        max_count = new_df[keyword].max()
                        min_count = new_df[keyword].min()
                        denom = max_count - min_count if max_count != min_count else 1
                        new_df[keyword] = (new_df[keyword] - min_count) / denom if denom else 0
                    
                    # Sum of normalized counts
                    new_df['Normalized_Count'] = new_df[st.session_state.final_selected_keywords].sum(axis=1)

                    # Sum of normalized counts for the document
                    df.at[index, 'Normalized_Count'] = new_df['Normalized_Count'].sum()


                sentence_df = pd.concat([sentence_df, new_df], ignore_index=True)


                # Update the progress bar
                progress = int((index + 1) / total_links * 100)
                progress_bar.progress(progress)

            except requests.RequestException:
                st.write(f"Error accessing {row['link']}")
        
        # Store processed data in session state
        st.session_state.df = df
        st.session_state.sentence_df = sentence_df
        st.session_state.processed = True      

        # Sort the dataframes based on the selected option
        df_sorted = apply_sorting(st.session_state.df, 'Normalized_Count' if sort_option == 'Overall' else sort_option)
        sentence_df_sorted = apply_sorting(st.session_state.sentence_df, 'Normalized_Count' if sort_option == 'Overall' else sort_option)

        # Display each row separately with hyperlinks and keyword info
        for index, row in df_sorted.iterrows():
            # Check if the row is selected in the multi-select box
            #if index in st.session_state.row_selection:
            st.markdown(f"### [{row['title']}]({row['link']})")
            # Display keyword counts on one line with | separator
            keyword_info = ' | '.join([f"{keyword}: {row[keyword]}" for keyword in st.session_state.final_selected_keywords])
            st.write(keyword_info)
            
            # Display top 5 sentences in an expander
            top_sentences = sentence_df_sorted[sentence_df_sorted['link'] == row['link']].head(5)
            with st.expander(f"Top Sentences for {row['title']}"):
                for _, sentence_row in top_sentences.iterrows():
                    text_content = sentence_row['sentence']
                    if translate:
                        # Assuming text_content contains the text extracted from the link
                        text_content = translate_text_with_google_cloud(text_content, "English")
                    st.write(text_content)

            # Delimiter between each item
            st.markdown("---")  # This creates a horizontal line as a delimiter

        # Multi-select box for row selection and dataframe update logic
        #row_ids = st.session_state.df.index.tolist()
        #st.session_state.row_selection = st.sidebar.multiselect('Select rows to include in further analysis:',
        #                                                        options=row_ids,
        #                                                        default=row_ids,
        #                                                        key=1)
        
        #st.session_state.first = False

    # Check if the data has been processed and stored in the session state
    #if 'processed' in st.session_state and st.session_state.processed and st.session_state.first is False:

def document_analysis():
    st.title("Run Document Analysis 📚")
    st.markdown("""
    Analyze the processed data in this final step. Utilize advanced AI models to summarize and extract key insights from the collected information. This section helps in understanding the broader context and significance of the data, providing valuable conclusions and takeaways from your research.
    """)

    # Function to calculate the cost estimate
    def calculate_cost_estimate(df, selected_model):
        encoding = tiktoken.encoding_for_model(selected_model)

        input_token_count = 0
        for link in df['link'].unique():
            top_sentences = df[df['link'] == link].nlargest(2, "Normalized_Count")
            extracts = "\n".join(top_sentences['sentence'])
            prompt = f"Your prompt text here: {extracts}"
            input_token_count += len(encoding.encode(prompt))

        # Define maximum output tokens based on the model
        max_output_tokens = 2000 if selected_model == "gpt-4" else 1000

        # Token pricing per 1K tokens (update as per current rates)
        pricing = {
            "gpt-4": (0.03, 0.06),
            "gpt-3.5-turbo-instruct": (0.0015, 0.002)
        }

        input_cost = (input_token_count / 1000) * pricing[selected_model][0]
        output_cost = (max_output_tokens / 1000) * pricing[selected_model][1] * len(df['link'].unique())

        total_cost = input_cost + output_cost
        return total_cost

    # Function to trim the prompt to fit within the token limit
    def trim_prompt_to_fit_limit(prompt, encoding, token_limit):
        tokens = encoding.encode(prompt)
        if len(tokens) > token_limit:
            # Trim tokens to fit within the limit, minus some buffer for the output
            buffer = 500  # Adjust buffer size as needed
            trimmed_tokens = tokens[:max(0, token_limit - buffer)]
            trimmed_prompt = encoding.decode(trimmed_tokens)
            return trimmed_prompt, True
        return prompt, False

    # Check if preprocessing has been done
    if 'sentence_df' not in st.session_state or 'df' not in st.session_state:
        st.warning("Please run preprocessing first.")
        return

    # Sidebar for GPT model selection
    models = ["gpt-3.5-turbo-instruct", "gpt-4"]
    selected_model = st.sidebar.selectbox("Select OpenAI Model:", models, index=0, key="model_select_key")

    st.dataframe(st.session_state.df)
    # Calculate and display the cost estimate
    if 'df' in st.session_state:
        cost_estimate = calculate_cost_estimate(st.session_state.sentence_df, selected_model)
        st.sidebar.write(f"Estimated cost for this run: ${cost_estimate:.2f}")


    # Initialize the progress bar
    progress_bar = st.sidebar.progress(0)

    # Get the total number of links to process for updating the progress bar
    total_links = len(st.session_state.df)

    if st.sidebar.button("Run Analysis"):
        all_summaries = []  # List to store individual summaries

        # For each link, process and summarize
        for index, link in enumerate(st.session_state.df['link'].unique()):
            top_sentences = st.session_state.sentence_df[st.session_state.sentence_df['link'] == link].nlargest(2, "Normalized_Count")
            extracts = "\n".join(top_sentences['sentence'])

            prompt = f"""I created a newsletter scraper that gives you got some non ordered extracts from longer documents:
            you are asked to write 1) "Brief Summary:" summarize the document in two sentences and 2) "Key numbers": one bullet point for each key number, and a description of what it is, iwth particular focus on monetary numbers
            below the extract from one document:\n{extracts}"""

            token_limit = 8000 if selected_model == "gpt-4" else 4000
            encoding = tiktoken.encoding_for_model(selected_model)

            trimmed_prompt, was_trimmed = trim_prompt_to_fit_limit(prompt, encoding, token_limit)


            # Call the OpenAI API
            if selected_model == "gpt-4":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=1000 
                )
                summary = response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    model=selected_model,
                    prompt=prompt,
                    max_tokens=1000 
                )
                summary = response.choices[0].text.strip()

            all_summaries.append(summary)

            # Update the progress bar
            progress = int((index + 1) / total_links * 100)
            progress_bar.progress(progress)

            st.session_state.all_summaries = all_summaries

        # Display individual summaries with hyperlinked titles
        with st.expander("See Individual Summaries"):
            for link, summary in zip(st.session_state.df['link'].unique(), all_summaries):
                if was_trimmed:
                    st.warning("The prompt was too long and had to be trimmed to fit within the token limit.")

                title = st.session_state.df[st.session_state.df['link'] == link]['title'].values[0]
                st.markdown(f"[**{title}**]({link})")  # Title as a hyperlink
                st.write(f"Summary: {summary}")
                st.markdown("---")  # separator
                
        # Combine all summaries for final summary
        combined_summaries = " ".join(all_summaries)
        final_prompt = f"Summarize the following summaries in 10 sentences with the following structure; 1) Summary 2) Key numbers:\n{combined_summaries}"

        token_limit = 8000 if selected_model == "gpt-4" else 4000
        encoding = tiktoken.encoding_for_model(selected_model)

        trimmed_prompt, was_trimmed = trim_prompt_to_fit_limit(final_prompt, encoding, token_limit)

        # Call the OpenAI API for final summary
        if selected_model == "gpt-4":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1000 
            )
            final_summary = response['choices'][0]['message']['content']
        else:
            response = openai.Completion.create(
                model=selected_model,
                prompt=final_prompt,
                max_tokens=2000
            )
            final_summary = response['choices'][0]['message']['content'] if selected_model == "gpt-4" else response.choices[0].text.strip()
            st.session_state.final_summary = final_summary
        
        # Display the final summary
        if was_trimmed:
            st.warning("The prompt was too long and had to be trimmed to fit within the token limit.")

        st.write(f"Final Summary: {final_summary}")

        if st.sidebar.button('Download Results'):
            output = BytesIO()

            # Create an Excel workbook and worksheet in memory
            workbook = xlsxwriter.Workbook(output, {'in_memory': True})
            
            # Writing final summary to the first sheet
            summary_sheet = workbook.add_worksheet('Summary')
            summary_sheet.write('A1', 'Final Summary')
            summary_sheet.write('A2', st.session_state.final_summary)
            
            # Writing one row per source to the second sheet
            sources_sheet = workbook.add_worksheet('Sources')
            sources_data = [{"ID": idx + 1, "Link": link, "Summary": summary} 
                            for idx, (link, summary) in enumerate(zip(st.session_state.df['link'].unique(), st.session_state.all_summaries))]
            sources_df = pd.DataFrame(sources_data)
            for idx, col_name in enumerate(sources_df.columns):
                sources_sheet.write(0, idx, col_name)
                for row_idx, value in enumerate(sources_df[col_name]):
                    sources_sheet.write(row_idx + 1, idx, value)
            
            # Writing one sheet per link
            for idx, link in enumerate(st.session_state.df['link'].unique(), 1):
                link_df = st.session_state.sentence_df[st.session_state.sentence_df['link'] == link]
                link_sheet = workbook.add_worksheet(f'Link {idx}')  # Using idx for sheet name instead of link
                for col_idx, col_name in enumerate(link_df.columns):
                    link_sheet.write(0, col_idx, col_name)
                    for row_idx, value in enumerate(link_df[col_name]):
                        link_sheet.write(row_idx + 1, col_idx, value)
            
            workbook.close()

            # Providing a download button for the generated Excel file
            st.sidebar.download_button(
                label="Download Excel workbook",
                data=output.getvalue(),
                file_name="analysis_results.xlsx",
                mime="application/vnd.ms-excel"
            ) 

    # Email sending feature
    smtp_user = os.environ.get('GMAIL_USER')
    smtp_password = os.environ.get('GMAIL_PASSWORD')
    def send_email(to_emails, subject, content):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = os.getenv('smtp_user')
        smtp_password = os.getenv('smtp_password')
        
        # Accessing all_summaries from st.session_state
        all_summaries = st.session_state.all_summaries

        # Create the message
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        # Handling both single and multiple recipient cases
        if isinstance(to_emails, list):
            msg['To'] = ', '.join(to_emails)
        else:
            msg['To'] = to_emails
        msg['Subject'] = subject

        # Create email body
        email_body = f"Final Summary:\n{content}\n\nDocument Summaries:\n"
        for link, summary in zip(st.session_state.df['link'].unique(), all_summaries):
            title = st.session_state.df[st.session_state.df['link'] == link]['title'].values[0]
            email_body += f"\n[**{title}**]({link})\nSummary: {summary}\n---"

        # Attach the content
        msg.attach(MIMEText(email_body, 'plain'))

        # Send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()

    # Email sending feature
    st.sidebar.write("Enter your email to receive the analysis:")
    user_email = st.sidebar.text_input("Email")

    if st.sidebar.button("Send Email"):
        if user_email:
            try:
                content = st.session_state.final_summary  # Assuming this is your final summary
                send_email([user_email, "darlain.edeme@iea.org"], "Document Analysis Results", content)
                st.sidebar.success("Email sent successfully!")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
        else:
            st.sidebar.error("Please enter a valid email and ensure the summary is generated.")

                
pages = {
    "🏠 Welcome": welcome_page,
    "🌍  Area Selection": area_selection,
    "✅ Selected Area Check ": selected_area_check,
    "🛠️ Define research": define_research,
    "🔍 Research": research,
    "💻 Pre-processing": run_preprocessing,
    "📚 Document Analysis": document_analysis

}


selection = st.sidebar.radio("Go to", list(pages.keys()))

# Sidebar information box
st.sidebar.info("If you have any issues, contact Darlain at darlain.edeme@iea.org")

# Run the selected page
pages[selection]()