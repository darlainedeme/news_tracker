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
    menu_options = ['Country', 'Continent', 'WEO Region', 'World', 'No specific area']
    selection = st.sidebar.selectbox("Choose a category", menu_options, index=0)
    if selection == 'Country':
        countries = data['field_3'].unique().tolist()
        selected_country = st.sidebar.multiselect('Choose countries', countries, default=['France'])
        subset_data = data[data['field_3'].isin(selected_country)]
    elif selection == 'Continent':
        continents = data['continent'].unique().tolist()
        selected_continent = st.sidebar.selectbox('Choose a continent', continents, index=continents.index('Europe'))
        subset_countries = data[data['continent'] == selected_continent]['field_3'].unique().tolist()
        selected_country = st.sidebar.multiselect('Choose countries', subset_countries, default=subset_countries)
        subset_data = data[data['field_3'].isin(selected_country)]
    elif selection == 'WEO Region':
        weo_regions = data['Code_Region'].unique().tolist()
        selected_weo = st.sidebar.selectbox('Choose a WEO Region', weo_regions, index=weo_regions.index('EUA'))
        subset_countries = data[data['Code_Region'] == selected_weo]['field_3'].unique().tolist()
        selected_country = st.sidebar.multiselect('Choose countries', subset_countries, default=subset_countries)
        subset_data = data[data['field_3'].isin(selected_country)]
    elif selection == 'World':
        subset_data = data
        selected_country = list(data.field_3)
    st.session_state.subset_data = subset_data
    st.session_state.selected_countries = selected_country

    # Read the CSV
    tld_data = pd.read_csv(os.path.join('data', 'tld.csv'), encoding='utf-8')
    # Extracting the TLDs based on selected countries
    selected_tlds = tld_data[tld_data['country'].isin(selected_country)]['tld'].tolist()
    
    st.session_state.country_codes = selected_tlds

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")
    folium.GeoJson(subset_data).add_to(m)
    folium_static(m) #, width=1500, height=800)

def selected_area_check():
    st.write("### Check the table below, and confirm it's the region you are interested in.")
    st.write("If it matches your criteria, proceed to the next step. Otherwise, return to the 'Area Selection' step to adjust your choices.")

    if 'subset_data' in st.session_state and st.session_state.subset_data is not None:
        st.table(st.session_state.subset_data[['field_3', 'continent', 'Code_Region']].rename(columns={'field_3': 'Country'}))
    else:
        st.write("No countries selected in the 'Area Selection' step.")

def define_research():
    st.title("Research Customization")
    
    # Load necessary data
    links_df = pd.read_csv('data/links.csv', encoding='utf-8')
    languages_df = pd.read_csv('data/languages.csv', encoding='utf-8')
    comp_keywords_df = pd.read_csv('data/complementary_keywords.csv', encoding='utf-8') # Load complementary keywords
        
    # 1. Kind of Research
    st.subheader("1. Research Type")
    st.write("Please select the type of research you are interested in.")
    research_type = st.radio("", ["policies", "news", "projects"])
    st.session_state.research_type = research_type

    # Separator
    st.markdown("---")

    # 2. Sources of Information
    st.subheader("2. Information Sources")
    st.write("Choose where you want to gather information from.")
    source_selection = st.radio("Choose a source:", ["predefined sources search", "general google search", "general twitter search", "general linkedin search"])  
    st.session_state.sources = source_selection

    # Limit research to country-specific domains
    if st.session_state.sources == "general google search":
        limit_to_country = st.checkbox("Limit research to country-specific domains?", value=True, help="This might provide precise results for country-specific information but will exclude international sources with .com, .org, etc.")
        st.session_state.limit_to_country = limit_to_country


    # 2.1 Official Sources (if selected)
    if source_selection == "predefined sources search":
        # Get unique types from links.csv
        types_list = links_df.loc[links_df['Country'].isin(st.session_state.selected_countries), 'Type'].unique().tolist()
        official_sources = st.multiselect("", types_list, default="Official Ministry website")
        st.session_state.official_sources = official_sources
        
        # Display a table of types and counts only for the selected countries
        source_counts = links_df[links_df['Country'].isin(st.session_state.selected_countries)].groupby(['Type', 'Country']).size().unstack(fill_value=0)
        st.write(source_counts)
        st.session_state.selected_predefined_links = list(links_df[ (links_df['Country'].isin(st.session_state.selected_countries)) & (links_df['Type'].isin(st.session_state.official_sources)) ].Link)
            
    # Separator
    st.markdown("---")

    # 3. Period of Interest
    st.subheader("3. Period of Interest")
    st.write("Select the timeframe for your research.")
    period_options = ["custom", "last 24h", "last week", "last two weeks", "last month", 
                      "last three months", "last 6 months", "last year", "last 2y",
                      "last 3y", "last 4y", "last 5y", "last 10y"]
    default_index = period_options.index('last week')
    selected_period = st.selectbox("", period_options, index=default_index)

    if selected_period == "custom":
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
    else:
        end_date = datetime.date.today()
        if selected_period == "last 24h":
            start_date = end_date - datetime.timedelta(days=1)
        elif selected_period == "last week":
            start_date = end_date - datetime.timedelta(weeks=1)
        elif selected_period == "last two weeks":
            start_date = end_date - datetime.timedelta(weeks=2)
        elif selected_period == "last month":
            start_date = end_date - datetime.timedelta(weeks=4)
        elif selected_period == "last three months":
            start_date = end_date - datetime.timedelta(weeks=12)
        elif selected_period == "last 6 months":
            start_date = end_date - datetime.timedelta(weeks=26)
        elif selected_period == "last year":
            start_date = end_date - datetime.timedelta(weeks=52)
        elif selected_period == "last 2y":
            start_date = end_date - datetime.timedelta(weeks=104)
        elif selected_period == "last 3y":
            start_date = end_date - datetime.timedelta(weeks=52*3)
        elif selected_period == "last 4y":
            start_date = end_date - datetime.timedelta(weeks=52*4)
        elif selected_period == "last 5y":
            start_date = end_date - datetime.timedelta(weeks=52*5)
        elif selected_period == "last 10y":
            start_date = end_date - datetime.timedelta(days=3650)
    
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    # Separator
    st.markdown("---")
    # 4. Language
    st.subheader("4. Language")
    # Filter languages based on the selected countries
    selected_country_languages = languages_df[languages_df['Country'].isin(st.session_state.selected_countries)]
    default_languages = selected_country_languages.melt(id_vars=['Country'], value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']).dropna()['value'].unique().tolist()
    
    # Set up multiselect with all available languages and defaults based on the selected countries
    all_languages = languages_df.melt(id_vars=['Country'], value_vars=['Language#1', 'Language#2', 'Language#3', 'Language#4']).dropna()['value'].unique().tolist()
    selected_language = st.multiselect("Language(s):", sorted(all_languages), default=default_languages)
    st.session_state.selected_language = selected_language

    # Separator
    st.markdown("---")

    # 5. Mandatory Keywords
    st.subheader("5. Mandatory Keywords")
    mandatory_keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
    selected_mandatory_keywords = st.multiselect("Mandatory Keywords:", sorted(mandatory_keywords_df['keyword'].tolist()), default=['electric vehicle'])
    st.session_state.selected_mandatory_keywords = selected_mandatory_keywords
    
    # Additional Mandatory Keywords
    additional_mandatory_keywords = st.text_area("Additional Mandatory Keywords (comma-separated):")
    if additional_mandatory_keywords:
        additional_mandatory_keywords = [kw.strip() for kw in additional_mandatory_keywords.split(",")]
        selected_mandatory_keywords.extend(additional_mandatory_keywords)
        st.session_state.selected_mandatory_keywords = selected_mandatory_keywords

    st.markdown("---")
    
    # 5. Topic Keywords
    st.subheader("6. Topic Keywords")
    keywords_df = pd.read_csv('data/keywords.csv', encoding='utf-8')
    
    # Filter out mandatory keywords from topic keywords list
    filtered_topic_keywords = [k for k in keywords_df['keyword'].tolist() if k not in selected_mandatory_keywords]
    
    selected_keywords = st.multiselect("Keywords:", sorted(filtered_topic_keywords))
    custom_keywords = st.text_input("Add additional topic keywords (comma separated):")
    if custom_keywords:
        custom_keywords_list = [keyword.strip() for keyword in custom_keywords.split(',')]
        selected_keywords.extend(custom_keywords_list)

    # Separator
    st.markdown("---")

    st.session_state.selected_keywords = selected_keywords
    
    selected_comp_keywords = []
       
    # 6. Complementary Research Keywords
    st.subheader("7. Complementary Research Keywords")
    
    custom_comp_keywords = st.text_input("Add additional complementary keywords (comma separated):")
    if custom_comp_keywords:
        custom_comp_keywords_list = [keyword.strip() for keyword in custom_comp_keywords.split(',')]
        selected_comp_keywords.extend(custom_comp_keywords_list)
           
    # st.session_state.include_monetary_info = False
    # Include Monetary Information Button
    include_monetary_info = st.checkbox("Include monetary information?")
           
    if include_monetary_info:    
        st.session_state.include_monetary_info
        # Open currencies.csv and get currencies and symbols for selected countries
        currencies_df = pd.read_csv('data/currencies.csv', encoding='utf-8')
        relevant_currencies = currencies_df.loc[currencies_df['country'].isin(st.session_state.selected_countries), ['currency_1', 'currency_1_symbol']].values.flatten()
        
        # Add the currency data to the complementary keywords list
        comp_keywords = sorted(comp_keywords_df['keyword'].tolist()) + list(relevant_currencies)
        
        add_comp_keywords = st.multiselect("Keywords:", comp_keywords,  default=comp_keywords)

        selected_comp_keywords.extend(add_comp_keywords)
        
    st.session_state.selected_comp_keywords = selected_comp_keywords

    # Extract respective translations for the selected keywords
    main_selected_translations = {}
    comp_selected_translations = {}
    mandatory_selected_translations = {}
    
    for language in selected_language:
        # Translations for mandatory keywords
        mandatory_selected_translations[language] = [
            mandatory_keywords_df.loc[mandatory_keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in mandatory_keywords_df['keyword'].tolist() else keyword 
            for keyword in selected_mandatory_keywords
        ]
        
        # Translations for main keywords
        main_selected_translations[language] = [
            keywords_df.loc[keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in keywords_df['keyword'].tolist() else keyword 
            for keyword in selected_keywords
        ]
    
        # Translations for complementary keywords if monetary info is selected
        if include_monetary_info:
            comp_selected_translations[language] = [
                comp_keywords_df.loc[comp_keywords_df['keyword'] == keyword, language].tolist()[0] if keyword in comp_keywords_df['keyword'].tolist() else keyword 
                for keyword in st.session_state.selected_comp_keywords
            ]

    # After:
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
    final_selected_keywords = list(selected_mandatory_keywords)
    final_selected_keywords.extend(st.session_state.selected_keywords)
    final_selected_keywords.extend(st.session_state.selected_comp_keywords)

    # if include_monetary_info and 'selected_comp_keywords' in st.session_state:
        # final_selected_keywords.extend(st.session_state.selected_comp_keywords)
    
    # Removing potential duplicates from selected keywords
    # final_selected_keywords = list(set(final_selected_keywords))
    
    # Add to session_state
    st.session_state.translated_trans_keywords = translated_trans_keywords
    st.session_state.final_selected_keywords = final_selected_keywords

    st.write(final_selected_keywords)

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
        if st.session_state.include_monetary_info:
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
            

def document_analysis():
    st.title("Run Document Analysis 📚")

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

    # Sidebar for GPT model selection
    models = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4"]
    selected_model = st.sidebar.selectbox("Select OpenAI Model:", models, index=0, key="model_select_key")

    # Initialize the progress bar
    progress_bar = st.sidebar.progress(0)

    # Get the total number of links to process for updating the progress bar
    total_links = len(df)

    if st.sidebar.button("Run Analysis"):
        # For each keyword, create a new column initialized to 0
        for keyword in st.session_state.final_selected_keywords:
            df[keyword] = 0
                           
        # Iterate through each link in the dataframe
        for index, (idx, row) in enumerate(df.iterrows()):
            try:
                response = requests.get(row['link'])
                response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
                
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
                # sentences = re.split(r'\n\s*\n', text_content)
                
                # Initialize an empty list to collect sentence data
                sentence_data_list = []
                
                for sentence_id, sentence in enumerate(sentences, 1):
                    sentence_data = {
                        'title': row['title'],
                        'link': row['link'],
                        'sentence_id': f"{index + 1}_{sentence_id}",
                        'sentence': sentence
                    }
                    # Ensure that keyword counts are stored as integers
                    for keyword, trans_keyword in zip(st.session_state.final_selected_keywords, st.session_state.translated_trans_keywords):
                        sentence_data[keyword] = sentence.count(trans_keyword.lower())
                    
                    # Add the sentence data to the list
                    sentence_data_list.append(sentence_data)
                
                # Create a new DataFrame from the list
                new_df = pd.DataFrame(sentence_data_list)
                
                # If sentence_df is empty (i.e., it's the first iteration), assign new_df to sentence_df
                # Otherwise, concatenate new_df with sentence_df
                if sentence_df.empty:
                    sentence_df = new_df
                else:
                    sentence_df = pd.concat([sentence_df.reset_index(drop=True), new_df], ignore_index=True)

                            
                    sentence_df['sentence'] = sentence_df['sentence'].str.replace('\n', ' ')
                    sentence_df['sentence'] = sentence_df['sentence'].apply(lambda x: re.sub(r'\s{2,}', '-', re.sub(r'\s+', ' ', x.replace('\n', ' ')))
)        
            
            except requests.RequestException:
                st.write(f"Error accessing {row['link']}")

            
                    
            # Update the progress bar
            progress = int((index + 1) / total_links * 100)
            progress_bar.progress(progress)
        
        # Function to normalize and sum keyword counts for a single sentence
        def normalize_and_sum(row):
            max_count = row.max()
            min_count = row.min()
            # Avoid division by zero
            denom = max_count - min_count if max_count != min_count else 1
            normalized_counts = (row - min_count) / denom
            return normalized_counts.sum()

        # Apply the function to each row of the keyword count columns
        sentence_df['Normalized_Count'] = sentence_df[st.session_state.final_selected_keywords].apply(normalize_and_sum, axis=1)
                
        all_summaries = []  # List to store individual summaries

        for link in df['link'].unique():
            top_sentences = sentence_df[sentence_df['link'] == link].nlargest(2, "Normalized_Count")
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

        # Combine all summaries
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
        
        final_summary = summary.replace('*', '&#42;').replace('_', '&#95;')
        st.write(f"Final Summary: {final_summary}")

        # Storing necessary variables in the session state
        st.session_state.sentence_df = sentence_df
        st.session_state.final_summary = final_summary
        st.session_state.all_summaries = all_summaries
        st.session_state.df = df 

        # Display individual summaries in an expander
        with st.expander("See Individual Summaries"):
            for link, summary in zip(df['link'].unique(), all_summaries):
                title = df[df['link'] == link]['title'].values[0]
                title = title.replace('*', '&#42;').replace('_', '&#95;')
                st.write(f"[{title}]({link})")
                summary = summary.replace('*', '&#42;').replace('_', '&#95;')
                st.write(f"Summary: {summary}")
                st.markdown("---")  # separator

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

pages = {
    "🌍  Area Selection": area_selection,
    "✅ Selected Area Check ": selected_area_check,
    "🛠️ Define research": define_research,
    "🔍 Research": research,
    "📚 Run Document Analysis": document_analysis

}

selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()

