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

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
cse_id = os.getenv('CSE_ID')
api_key = os.getenv('API_KEY')
            
data = gpd.read_file(os.path.join('data', 'merged_file.gpkg'))
data = data[data['field_3'].notna()]

def area_selection():
    # Initialize session state variables if they don't exist
    if 'selected_countries' not in st.session_state:
        st.session_state.selected_countries = []
    if 'subset_data' not in st.session_state:
        st.session_state.subset_data = None
    if 'country_codes' not in st.session_state:
        st.session_state.country_codes = []

    menu_options = ['Country', 'Continent', 'WEO Region', 'World', 'No specific area']
    selection = st.sidebar.selectbox("Choose a category", menu_options, index=0)

    data = gpd.read_file(os.path.join('data', 'merged_file.gpkg'))
    data = data[data['field_3'].notna()]

    if selection == 'Country':
        countries = data['field_3'].unique().tolist()
        selected_countries = st.sidebar.multiselect('Choose countries', countries, default=st.session_state.selected_countries)
        st.session_state.selected_countries = selected_countries
        subset_data = data[data['field_3'].isin(selected_countries)]

    elif selection == 'Continent':
        continents = data['continent'].unique().tolist()
        selected_continent = st.sidebar.selectbox('Choose a continent', continents, index=continents.index('Europe'))
        subset_countries = data[data['continent'] == selected_continent]['field_3'].unique().tolist()
        selected_countries = st.sidebar.multiselect('Choose countries', subset_countries, default=subset_countries)
        subset_data = data[data['field_3'].isin(selected_countries)]
        st.session_state.selected_countries = selected_countries

    elif selection == 'WEO Region':
        weo_regions = data['Code_Region'].unique().tolist()
        selected_weo = st.sidebar.selectbox('Choose a WEO Region', weo_regions, index=weo_regions.index('EUA'))
        subset_countries = data[data['Code_Region'] == selected_weo]['field_3'].unique().tolist()
        selected_countries = st.sidebar.multiselect('Choose countries', subset_countries, default=subset_countries)
        subset_data = data[data['field_3'].isin(selected_countries)]
        st.session_state.selected_countries = selected_countries

    elif selection == 'World':
        subset_data = data
        st.session_state.selected_countries = list(data.field_3)

    st.session_state.subset_data = subset_data

    # Read the CSV
    tld_data = pd.read_csv(os.path.join('data', 'tld.csv'), encoding='utf-8')
    # Extracting the TLDs based on selected countries
    selected_tlds = tld_data[tld_data['country'].isin(st.session_state.selected_countries)]['tld'].tolist()
    
    st.session_state.country_codes = selected_tlds

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")
    folium.GeoJson(subset_data).add_to(m)
    folium_static(m)


def selected_area_check():
    st.write("### Check the table below, and confirm it's the region you are interested in.")
    st.write("If it matches your criteria, proceed to the next step. Otherwise, return to the 'Area Selection' step to adjust your choices.")

    if 'subset_data' in st.session_state and st.session_state.subset_data is not None:
        st.table(st.session_state.subset_data[['field_3', 'continent', 'Code_Region']].rename(columns={'field_3': 'Country'}))
    else:
        st.write("No countries selected in the 'Area Selection' step.")

def define_research():
    st.title("Research Customization")

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

    # Load necessary data
    links_df = pd.read_csv('data/links.csv', encoding='utf-8')
    languages_df = pd.read_csv('data/languages.csv', encoding='utf-8')
    comp_keywords_df = pd.read_csv('data/complementary_keywords.csv', encoding='utf-8')
    mandatory_keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
    keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')

    # 1. Kind of Research
    st.subheader("1. Research Type")
    st.session_state.research_type = st.radio("", ["policies", "news", "projects"], index=["policies", "news", "projects"].index(st.session_state.research_type))

    # Separator
    st.markdown("---")

    # 2. Sources of Information
    st.subheader("2. Information Sources")
    st.session_state.sources = st.radio("Choose a source:", ["predefined sources search", "general google search", "general twitter search", "general linkedin search"], index=["predefined sources search", "general google search", "general twitter search", "general linkedin search"].index(st.session_state.sources))

    # Limit research to country-specific domains
    if st.session_state.sources == "general google search":
        st.session_state.limit_to_country = st.checkbox("Limit research to country-specific domains?", value=st.session_state.limit_to_country, help="This might provide precise results for country-specific information but will exclude international sources with .com, .org, etc.")

    # 2.1 Official Sources (if selected)
    if st.session_state.sources == "predefined sources search":
        types_list = links_df.loc[links_df['Country'].isin(st.session_state.selected_countries), 'Type'].unique().tolist()
        st.session_state.official_sources = st.multiselect("", types_list, default=st.session_state.official_sources)
        
        source_counts = links_df[links_df['Country'].isin(st.session_state.selected_countries)].groupby(['Type', 'Country']).size().unstack(fill_value=0)
        st.write(source_counts)
        st.session_state.selected_predefined_links = list(links_df[(links_df['Country'].isin(st.session_state.selected_countries)) & (links_df['Type'].isin(st.session_state.official_sources))].Link)
            
    # Separator
    st.markdown("---")

    # 3. Period of Interest
    st.subheader("3. Period of Interest")
    period_options = ["custom", "last 24h", "last week", "last two weeks", "last month", 
                      "last three months", "last 6 months", "last year", "last 2y",
                      "last 3y", "last 4y", "last 5y", "last 10y"]
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = 'last week'
    st.session_state.selected_period = st.selectbox("", period_options, index=period_options.index(st.session_state.selected_period))

    if st.session_state.selected_period == "custom":
        st.session_state.start_date = st.date_input("Start Date", value=st.session_state.start_date)
        st.session_state.end_date = st.date_input("End Date", value=st.session_state.end_date)
    else:
        st.session_state.end_date = datetime.date.today()
        if st.session_state.selected_period == "last 24h":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(days=1)
        elif st.session_state.selected_period == "last week":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=1)
        elif st.session_state.selected_period == "last two weeks":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=2)
        elif st.session_state.selected_period == "last month":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=4)
        elif st.session_state.selected_period == "last three months":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=12)
        elif st.session_state.selected_period == "last 6 months":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=26)
        elif st.session_state.selected_period == "last year":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=52)
        elif st.session_state.selected_period == "last 2y":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=104)
        elif st.session_state.selected_period == "last 3y":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=156)
        elif st.session_state.selected_period == "last 4y":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=208)
        elif st.session_state.selected_period == "last 5y":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(weeks=260)
        elif st.session_state.selected_period == "last 10y":
            st.session_state.start_date = st.session_state.end_date - datetime.timedelta(days=3650)

    # Separator
    st.markdown("---")

    # 4. Language
    st.subheader("4. Language")
    selected_country_languages = languages_df[languages_df['Country'].isin(st.session_state.selected_countries)]
    default_languages = selected_country_languages.melt(id_vars=['Country'], value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']).dropna()['value'].unique().tolist()
    all_languages = languages_df.melt(id_vars=['Country'], value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']).dropna()['value'].unique().tolist()
    st.session_state.selected_language = st.multiselect("Language(s):", sorted(all_languages), default=default_languages)

    # Separator
    st.markdown("---")

    # 5. Mandatory Keywords
    st.subheader("5. Mandatory Keywords")
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

    st.session_state.selected_mandatory_keywords = st.multiselect(
        "Mandatory Keywords:", 
        all_mandatory_keywords, 
        default=st.session_state.selected_mandatory_keywords
    )

    st.markdown("---")
    
    # 6. Topic Keywords
    st.subheader("6. Topic Keywords")
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
    st.session_state.selected_keywords = st.multiselect("Keywords:", filtered_topic_keywords, default=st.session_state.selected_keywords)

    st.markdown("---")

    # 7. Complementary Research Keywords
    st.subheader("7. Complementary Research Keywords")
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

    st.session_state.selected_comp_keywords = st.multiselect("Keywords:", all_comp_keywords, default=st.session_state.selected_comp_keywords)

    # Extract respective translations for the selected keywords
    main_selected_translations = {}
    comp_selected_translations = {}
    mandatory_selected_translations = {}

    for language in st.session_state.selected_language:
        # Translations for mandatory keywords
        mandatory_selected_translations[language] = [
            mandatory_keywords_df.loc[mandatory_keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in mandatory_keywords_df['keyword'].tolist() else keyword 
            for keyword in st.session_state.selected_mandatory_keywords
        ]

        # Translations for main keywords
        main_selected_translations[language] = [
            keywords_df.loc[keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in keywords_df['keyword'].tolist() else keyword 
            for keyword in st.session_state.selected_keywords
        ]

        # Translations for complementary keywords
        comp_selected_translations[language] = [
            comp_keywords_df.loc[comp_keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in comp_keywords_df['keyword'].tolist() else keyword 
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

    # Ensure that the necessary data is in the session state
    if 'final_selected_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return
    
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
    
    
    
    if st.sidebar.button("Run Research"):
        links_list = []
        # Clear previous results
        query = construct_query()
         
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
    
        total_results = int(response.json().get("searchInformation", {}).get("totalResults", 0))
        # URL encode the query string
        encoded_query = urllib.parse.quote_plus(query)  
        
        # Create the hyperlink URL for the query on Google
        google_search_url = f"https://www.google.com/search?q={encoded_query}"
    
        # Display the estimated total number of search results and the link
        st.markdown(f"Total estimated results on Google: [{total_results}]({google_search_url})")
    
        # Reset the session_state for results to ensure new results overwrite previous ones
        st.session_state.results = []
        
        # Store the results in session_state, overwriting any previous results
        st.session_state.results = results
    
        # Display the results
        total_characters = 0  # Initialize a counter
        for result in results:
            st.subheader(f"[{result['title']}]({result['link']})")
            st.write(f"Source: {result['displayLink']}")
            links_list.append(result['displayLink'])
            
# =============================================================================
#             try:
#                 st.write(f"Format: {result['fileFormat']}")
#             except:
#                 continue
# =============================================================================
            st.write(f"Published Date: {result.get('pagemap', {}).get('metatags', [{}])[0].get('og:updated_time', 'N/A')}")
            snippet_length = len(result.get('snippet', ''))
            st.write(f"Length in Characters: {snippet_length}")
            total_characters += snippet_length  # Update the counter
            
            # Summarize using gensim's TextRank
            summary = result['snippet']
            st.write(f"Summary: {summary}")
            
            # st.write(result)
            st.markdown("---")  # separator
        
        st.write(links_list)
        
        # Save the accumulated value to st.session_state
        st.session_state.total_characters = total_characters
        
        # Display the total character count
        st.write(f"Total Characters in all Snippets: {total_characters}")

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
    st.title("Run Preprocessing 🔄")

    # Ensure that the necessary data is in the session state
    if 'final_selected_keywords' not in st.session_state or 'translated_trans_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return

    # Load the CSV
    df = pd.read_csv(st.session_state.filename, encoding='utf-8')
    df = df[0:2]

    # Display the number of links in the sidebar
    st.sidebar.write(f"Total Links: {len(df)}")

    # Create an empty dataframe for sentence-level results
    sentence_df = pd.DataFrame(columns=['title', 'link', 'sentence_id', 'sentence'] + st.session_state.final_selected_keywords)

    if st.sidebar.button("Run Preprocessing"):
        # For each keyword, create a new column initialized to 0
        for keyword in st.session_state.final_selected_keywords:
            df[keyword] = 0

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
                sentence_df = pd.concat([sentence_df, new_df], ignore_index=True)

            except requests.RequestException:
                st.write(f"Error accessing {row['link']}")

        # Storing necessary variables in the session state
        st.session_state.sentence_df = sentence_df
        st.session_state.df = df

        st.write("Preprocessing completed successfully.")

def document_analysis():
    st.title("Run Document Analysis 📚")

    # Check if preprocessing has been done
    if 'sentence_df' not in st.session_state or 'df' not in st.session_state:
        st.warning("Please run preprocessing first.")
        return

    # Sidebar for GPT model selection
    models = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4"]
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
                    messages=messages
                )
                summary = response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    model=selected_model,
                    prompt=prompt,
                    max_tokens=100
                )
                summary = response.choices[0].text.strip()

            all_summaries.append(summary)

            # Update the progress bar
            progress = int((index + 1) / total_links * 100)
            progress_bar.progress(progress)

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
            final_summary = response.choices[0].text.strip()
        
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


pages = {
    "🌍  Area Selection": area_selection,
    "✅ Selected Area Check ": selected_area_check,
    "🛠️ Define research": define_research,
    "🔍 Research": research,
    "🔍 Pre-processing": run_preprocessing,
    "📚 Document Analysis": document_analysis

}


selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()

