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

# logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
cse_id = os.getenv('CSE_ID')
api_key = os.getenv('API_KEY')

data = gpd.read_file(os.path.join('data', 'merged_file.gpkg'))
data = data[data['field_3'].notna()]

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
    selection = st.sidebar.selectbox("Choose a category", menu_options, index=0)

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
    st.markdown("""
    Here, you can review the geographical areas you have selected. This step is crucial to ensure that the regions of interest are correctly chosen before proceeding with the research. If the selected areas don't match your criteria, you can return to the 'Area Selection' step to adjust your choices.
    """)

    st.write("### Check the table below, and confirm it's the region you are interested in.")
    st.write("If it matches your criteria, proceed to the next step. Otherwise, return to the 'Area Selection' step to adjust your choices.")

    if 'subset_data' in st.session_state and st.session_state.subset_data is not None:
        st.table(st.session_state.subset_data[['field_3', 'continent', 'Code_Region']].rename(columns={'field_3': 'Country'}))
    else:
        st.write("No countries selected in the 'Area Selection' step.")

def define_research():
    st.title("Research Customization")
    st.markdown("""
    Customize your research parameters in this section. You can select the type of research (policies, news, projects), choose information sources, set languages, and define keywords. These parameters will guide the data collection process, ensuring that the research is tailored to your specific needs and interests.
    """)

    # Ensure that the necessary data is in the session state
    if 'subset_data' not in st.session_state and st.session_state.subset_data is not None:
        st.warning("Please complete the previous steps first.")
        return
    
    # Initialize session state variables if not already set
    if 'research_type' not in st.session_state:
        st.session_state.research_type = 'policies'
    if 'sources' not in st.session_state:
        st.session_state.sources = 'predefined sources search'
    if 'limit_to_country' not in st.session_state:
        st.session_state.limit_to_country = True
    if 'official_sources' not in st.session_state:
        st.session_state.official_sources = []
    if 'selected_predefined_links' not in st.session_state:
        st.session_state.selected_predefined_links = []
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=1)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.date.today()
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = []
    if 'selected_mandatory_keywords' not in st.session_state:
        st.session_state.selected_mandatory_keywords = []
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []
    if 'selected_comp_keywords' not in st.session_state:
        st.session_state.selected_comp_keywords = []
    if 'include_monetary_info' not in st.session_state:
        st.session_state.include_monetary_info = False

    # Ensure that session state for research_type is initialized
    if 'research_type' not in st.session_state:
        st.session_state.research_type = "spendings"  # default value

    # 1. Kind of Research
    st.subheader("1. Research Type")
    st.session_state.research_type = st.radio(
        "",
        ["policies", "news", "spendings", "projects"],
        index=["policies", "news", "spendings", "projects"].index(st.session_state.research_type),
        help="Select the type of research you're interested in. Options include policies, news, and projects."
    )

    # Load necessary data based on the selected research type
    if st.session_state.research_type == "spendings":
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
        all_mandatory_keywords = list(set(list(all_mandatory_keywords) + list(st.session_state.selected_mandatory_keywords)))
        
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
                                                                  help="Select mandatory keywords. These are essential terms that must appear in the research results.")


    st.markdown("---")
    
    # 5. Topic Keywords
    st.subheader("5. Topic Keywords")
    keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
    all_topic_keywords = sorted(set(keywords_df['keyword'].tolist()))

    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []

    if 'selected_keywords' in st.session_state:
        all_topic_keywords = list(set(list(all_topic_keywords) + list(st.session_state.selected_keywords)))
        
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
                                                        help="Choose your topic keywords. These are the main terms related to your research topic.")

    st.markdown("---")

    # 6. Complementary Research Keywords
    st.subheader("6. Complementary Research Keywords")
    comp_keywords_df = pd.read_csv('data/complementary_keywords.csv', encoding='utf-8')
    all_comp_keywords = sorted(set(comp_keywords_df['keyword'].tolist()))

    if 'selected_comp_keywords' not in st.session_state:
        st.session_state.selected_comp_keywords = []

    if 'selected_comp_keywords' in st.session_state:
        all_comp_keywords = list(set(list(all_comp_keywords) + list(st.session_state.selected_comp_keywords)))
        
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
                                                             help="Select complementary keywords. These are additional terms that can enhance your research scope.")


    def translate_word(word, selected_language):
        """
        Translates a given word into the specified language.

        Args:
        word (str): The word to be translated.
        selected_language (str): The target language for translation.

        Returns:
        str: The translated word.
        """

        # Constructing the prompt for translation
        translation_prompt = f"Translate the following word into {selected_language}: {word}. GIVE ME AS AN OUTPUT ONLY THE TRANSLATED WORD. FOR TECHNICAL WORD ASSUME THE BEST TRANSLATION IN THE ENERGY SECTOR"

        # Preparing the messages for the API call
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": translation_prompt}
        ]

        # Making the API call
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=100
        )

        # Extracting the translation from the response
        translated_word = response['choices'][0]['message']['content']

        return translated_word

    # Extract respective translations for the selected keywords
    main_selected_translations = {}
    comp_selected_translations = {}
    mandatory_selected_translations = {}

    for language in st.session_state.selected_language:
        # Translations for mandatory keywords
        mandatory_selected_translations[language] = [
            mandatory_keywords_df.loc[mandatory_keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in mandatory_keywords_df['keyword'].tolist() else translate_word(keyword, st.session_state.selected_language[0]) 
            for keyword in st.session_state.selected_mandatory_keywords
        ]

        # Translations for main keywords
        main_selected_translations[language] = [
            keywords_df.loc[keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in keywords_df['keyword'].tolist() else translate_word(keyword, st.session_state.selected_language[0]) 
            for keyword in st.session_state.selected_keywords
        ]

        # Translations for complementary keywords
        comp_selected_translations[language] = [
            comp_keywords_df.loc[comp_keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in comp_keywords_df['keyword'].tolist() else translate_word(keyword, st.session_state.selected_language[0]) 
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
    st.title("Research 📚")
    st.markdown("""
    Conduct your research based on the parameters set in the previous steps. This section utilizes custom search engines to gather data and information from various sources. You can review the total results, check individual entries, and gather insights relevant to your selected areas and research criteria.
    """)

    # Ensure that the necessary data is in the session state
    if 'final_selected_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return

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

    def construct_query():
        query_parts = []
    
        # Incorporate translations into the main keywords query
        main_translations_query_parts = []
        for language, translations in st.session_state.main_selected_translations.items():
            translations_query = " OR ".join([f'"{translation}"' for translation in translations])
            if translations_query:  # ensure the list isn't empty
                main_translations_query_parts.append(translations_query)
        if main_translations_query_parts:
            query_parts.append(f"({' OR '.join(main_translations_query_parts)})")
        
        # Mandatory keywords need to be present, so we'll include them in the query
        mandatory_translations_query_parts = []
        for language, translations in st.session_state.mandatory_selected_translations.items():
            mandatory_translations_query = " AND ".join([f'"{translation}"' for translation in translations])
            if mandatory_translations_query:  # ensure the list isn't empty
                mandatory_translations_query_parts.append(mandatory_translations_query)
        if mandatory_translations_query_parts:
            query_parts.append(f"({' AND '.join(mandatory_translations_query_parts)})")
        
# =============================================================================
#         # Incorporate selected countries into the mandatory keywords
#         if 'selected_countries' in st.session_state:
#             countries_query = " AND ".join([f'"{country}"' for country in st.session_state.selected_countries])
#             if countries_query:  # ensure the list isn't empty
#                 query_parts.append(f"({countries_query})")
# =============================================================================
        
        # If the monetary information checkbox is selected, include complementary keywords in the query
        
        comp_translations_query_parts = []
        for language, translations in st.session_state.comp_selected_translations.items():
            comp_translations_query = " OR ".join([f'"{translation}"' for translation in translations])
            if comp_translations_query:  # ensure the list isn't empty
                comp_translations_query_parts.append(comp_translations_query)
        if comp_translations_query_parts:
            query_parts.append(f"({' OR '.join(comp_translations_query_parts)})")
    
        # Include date range
        start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')
        date_query = f'after:{start_date_str} before:{end_date_str}'
        query_parts.append(date_query)
    
        # Depending on the research source:
        if st.session_state.sources == "predefined sources search":
            predefined_sites_query = " OR ".join([f"site:{site}" for site in st.session_state.selected_predefined_links])
            query_parts.append(f"({predefined_sites_query})")
        elif st.session_state.sources == "general google search":
            # Excluding specific domains
            excluded_sites_query = " -site:iea.org -site:iea.blob.core.windows.net -site:en.wikipedia.org -site:irena.org"
            query_parts.append(excluded_sites_query)
            # If the option to limit the search to country-specific domains is selected
            if st.session_state.limit_to_country and 'selected_countries' in st.session_state:
                country_specific_sites_query = " OR ".join([f"site:{code.lower()}" for code in st.session_state.country_codes])
                query_parts.append(f"({country_specific_sites_query})")
        elif st.session_state.sources == "general twitter search":
            query_parts.append('site:twitter.com')
        elif st.session_state.sources == "general linkedin search":
            query_parts.append('site:linkedin.com')
    
        # Join the query parts using ' AND ' separator
        return ' AND '.join(query_parts)
    
    # Create a sidebar with a checkbox
    exact_keywords = st.sidebar.checkbox('Do you want exact keywords?', value=True)
    query = construct_query()

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

    # Modify the query based on the research type selected
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

        # Function to summarize content
        def summarize_content(content, max_length=2000):
            inputs = tokenizer.encode("summarize: " + content, return_tensors="pt", max_length=3000, truncation=True)
            summary_ids = model.generate(inputs, max_length=max_length, min_length=1000, length_penalty=0, num_beams=10, early_stopping=False)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        # Function to sort sentences based on the count of different keywords
        def sort_sentences(sentences, keywords):
            def count_unique_keywords(sentence):
                word_counts = Counter(word.lower() for word in sentence.split() if word.lower() in keywords)
                return len(word_counts)

            sorted_sentences = sorted(sentences, key=count_unique_keywords, reverse=True)
            return sorted_sentences
        
        def extract_metadata_and_index(pdf):
            title = pdf.metadata.get('title', 'No Title Found')
            index_content = ''

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    if 'content' in text.lower() or 'index' in text.lower() or 'table' in text.lower():
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

    # Checkbox for translation
    want_translation = st.sidebar.checkbox('Do you want to translate to English?', value=False)

    if want_translation:
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
            params = {
                'q': text,
                'target': target_language,
                'format': 'text',
                'key': api_key
            }
            response = requests.post(url, params=params)
            if response.status_code == 200:
                result = response.json()
                translated_text = result['data']['translations'][0]['translatedText']
                return translated_text
            else:
                raise Exception(f"Google Cloud Translation API error: {response.text}")

    if st.sidebar.button("Run Research"):
        links_list = []
        # Clear previous results
         
        # Define the endpoint URL (replace with your own endpoint and API key)
        url = "https://www.googleapis.com/customsearch/v1"
        results = []  # Store all results across pages
        for start_index in range(1, 101, 10):  # Start indices for pagination
            params = {
                'q': query,
                'cx': cse_id,
                'key': api_key,
                'num': 10,
                'start': start_index
            }
            response = requests.get(url, params=params)
            results.extend(response.json().get("items", []))
    
        total_results = len(results)
        # URL encode the query string
        encoded_query = urllib.parse.quote_plus(query)  
        
        # Create the hyperlink URL for the query on Google
        google_search_url = f"https://www.google.com/search?q={encoded_query}"
    
        # Display the estimated total number of search results and the link
        st.markdown(f"Total estimated results: [{total_results}]({google_search_url})")
    
        # Reset the session_state for results to ensure new results overwrite previous ones
        st.session_state.results = []
        
        # Store the results in session_state, overwriting any previous results
        st.session_state.results = results


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

                    # Example: Extract text from specific tags
                    main_content = soup.find('main')  # or soup.find('article') or another tag that usually contains the main text
                    if main_content:
                        text = ' '.join(p.get_text() for p in main_content.find_all('p'))
                    else:
                        text = soup.get_text()

                    # Summarize the webpage content
                    summary = summarize_content(text, max_length=100)  # Adjust max_length as needed

                    # Translate the summary if needed
                    if want_translation:
                        summary = translate_text_with_google_cloud(summary, st.session_state.selected_language[0])

                    # Display the summary in an expander
                    with st.expander("Show Summary"):
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

                                    # Translate extracted sentences if needed
                                    if want_translation:
                                        sorted_sentences = [translate_text_with_google_cloud(sentence, st.session_state.selected_language[0]) for sentence in sorted_sentences]

                                    st.write("Extracted Sentences:")
                                    for sentence in sorted_sentences:
                                        st.text(sentence)
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

        # Store the filename in session state for download
        st.session_state.filename = filename
        
        class PDF(FPDF):
            def header(self):
                # Arial bold 15
                self.set_font('Arial', 'B', 15)
                # Title
                self.cell(0, 10, 'Webpage Content', 0, 1, 'C')
        
            def footer(self):
                # Position at 1.5 cm from bottom
                self.set_y(-15)
                # Arial italic 8
                self.set_font('Arial', 'I', 8)
                # Page number
                self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')
        
        def download_links_as_pdfs(links, folder_path):
            # Ensure the directory exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            i = 0
            for link in links:
                i += 1
                if not link.startswith(("http://", "https://")):
                    link = "https://" + link
                    
                try:
                    response = requests.get(link)
                    soup = BeautifulSoup(response.content, 'html.parser')
        
                    # Remove all script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    text_content = soup.stripped_strings
                    content = "\n".join(text_content)
        
                    # Only generate PDF if content exists
                    if content:
                        pdf = PDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))  # Handle encoding issues
        
                        pdf_name = os.path.join(folder_path, str(i) + "_" + link.split('//')[-1].replace("/", "_") + ".pdf")
                        pdf.output(pdf_name)
                    else:
                        print(f"No content found for link {link}")
        
                except Exception as e:
                    print(f"Failed to process link {link}. Error: {e}")
            
        folder_path = "results/results"
        # download_links_as_pdfs(links_list, folder_path)

def run_preprocessing():
    st.title("Run Preprocessing 💻")
    st.markdown("""
    Preprocess the gathered data for analysis in this step. This involves data cleaning, normalization, and preparation for detailed analysis. The process ensures that the data is in the right format and structure, enabling effective and accurate analysis in the next steps.
    """)

    # Ensure that the necessary data is in the session state
    if 'final_selected_keywords' not in st.session_state or 'translated_trans_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return

    # Load the CSV
    df = pd.read_csv(st.session_state.filename, encoding='utf-8')
    # df = df[0:2]

    # Display the number of links in the sidebar
    st.sidebar.write(f"Total Links: {len(df)}")

    # Create an empty dataframe for sentence-level results
    sentence_df = pd.DataFrame(columns=['title', 'link', 'sentence_id', 'sentence'] + st.session_state.final_selected_keywords)

    if st.sidebar.button("Run Preprocessing"):
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

        # Storing necessary variables in the session state
        st.session_state.sentence_df = sentence_df
        st.session_state.df = df

        # Sorting the columns
        keyword_columns = sorted([col for col in df.columns if col not in ['title', 'link', 'word_count', 'Normalized_Count']])
        df = df[['title', 'word_count','Normalized_Count'] + keyword_columns + ['link']]

        # Display the results in a table
        st.write("Preprocessing completed successfully. Here are the results:")
        st.dataframe(df)

    # Multi-select box in the sidebar for row selection
    row_ids = df.index.tolist()
    row_selection = st.sidebar.multiselect('Select rows to include in further analysis:',
                                           options=row_ids,
                                           default=row_ids)

    # Directly update df by removing unselected rows
    df = df.loc[row_selection]

    # Display the updated dataframe
    st.write("Dataframe updated based on selected rows:")
    st.dataframe(df)

    # Update session state
    st.session_state.df = df

def document_analysis():
    st.title("Run Document Analysis 📚")
    st.markdown("""
    Analyze the processed data in this final step. Utilize advanced AI models to summarize and extract key insights from the collected information. This section helps in understanding the broader context and significance of the data, providing valuable conclusions and takeaways from your research.
    """)

    # Check if preprocessing has been done
    if 'sentence_df' not in st.session_state or 'df' not in st.session_state:
        st.warning("Please run preprocessing first.")
        return

    # Sidebar for GPT model selection
    models = ["gpt-3.5-turbo-instruct", "gpt-4"]
    selected_model = st.sidebar.selectbox("Select OpenAI Model:", models, index=0, key="model_select_key")

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
            you are asked to draft a brief summary of its content (two sentences) and all key numbers in it explained of each
            to be then inserted in the newsletter email. below the extract from one document: please max 100 words:\n{extracts}"""

            # Call the OpenAI API
            if selected_model == "gpt-4":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=4097 
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

        # Display individual summaries with hyperlinked titles
        with st.expander("See Individual Summaries"):
            for link, summary in zip(st.session_state.df['link'].unique(), all_summaries):
                title = st.session_state.df[st.session_state.df['link'] == link]['title'].values[0]
                st.markdown(f"[**{title}**]({link})")  # Title as a hyperlink
                st.write(f"Summary: {summary}")
                st.markdown("---")  # separator
                
        # Combine all summaries for final summary
        combined_summaries = " ".join(all_summaries)
        final_prompt = f"Summarize the following summaries in 10 sentences:\n{combined_summaries}"

        # Call the OpenAI API for final summary
        if selected_model == "gpt-4":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            final_summary = response['choices'][0]['message']['content']
        else:
            response = openai.Completion.create(
                model=selected_model,
                prompt=final_prompt,
                max_tokens=200
            )
            final_summary = response['choices'][0]['message']['content'] if selected_model == "gpt-4" else response.choices[0].text.strip()
            st.session_state.final_summary = final_summary
        
        # Display the final summary
        st.write(f"Final Summary: {final_summary}")

        # Download button for results
        if st.sidebar.button('Download Results'):
            output = BytesIO()
            workbook = xlsxwriter.Workbook(output, {'in_memory': True})
            # [Excel workbook creation logic here...]
            workbook.close()

            # Download button
            st.sidebar.download_button(
                label="Download Excel workbook",
                data=output.getvalue(),
                file_name="analysis_results.xlsx",
                mime="application/vnd.ms-excel"
            )       

    # Email sending feature
    smtp_user = os.environ.get('GMAIL_USER')
    smtp_password = os.environ.get('GMAIL_PASSWORD')
    def send_email(to_email, subject, content):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = os.getenv('smtp_user')
        smtp_password = os.getenv('smtp_password')

        # Create the message
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the content
        msg.attach(MIMEText(content, 'plain'))

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
                send_email(user_email, "Document Analysis Results", "ciao")
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