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

# Set your OpenAI API key
from dotenv import load_dotenv
load_dotenv()

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
        selected_country = st.sidebar.multiselect('Choose countries', countries, default=['Italy'])
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
    period_options = ["last 24h", "last week", "last two weeks", "last month", 
                      "last three months", "last 6 months", "last year", "last 2y",
                      "last 3y", "last 4y", "last 5y", "last 10y", "custom"]
    default_index = period_options.index('last 24h')
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
    selected_mandatory_keywords = st.multiselect("Mandatory Keywords:", sorted(mandatory_keywords_df['keyword'].tolist()))
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
    st.session_state.selected_keywords = selected_keywords
    
    # Separator
    st.markdown("---")
    

    selected_comp_keywords = []
    
   
    # 6. Complementary Research Keywords
    st.subheader("7. Complementary Research Keywords")
    
    custom_comp_keywords = st.text_input("Add additional complementary keywords (comma separated):")
    if custom_comp_keywords:
        custom_comp_keywords_list = [keyword.strip() for keyword in custom_comp_keywords.split(',')]
        selected_comp_keywords.extend(custom_comp_keywords_list)
    
    st.session_state.include_monetary_info = False
    # Include Monetary Information Button
    include_monetary_info = st.checkbox("Include monetary information?")

    if include_monetary_info:    
        st.session_state.include_monetary_info
        # Open currencies.csv and get currencies and symbols for selected countries
        currencies_df = pd.read_csv('data/currencies.csv', encoding='utf-8')
        relevant_currencies = currencies_df.loc[currencies_df['country'].isin(st.session_state.selected_countries), ['currency_1', 'currency_1_symbol']].values.flatten()
        
        # Add the currency data to the complementary keywords list
        comp_keywords = sorted(comp_keywords_df['keyword'].tolist()) + list(relevant_currencies)
        
        selected_comp_keywords = st.multiselect("Keywords:", comp_keywords,  default=comp_keywords)
    
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
    final_selected_keywords = list(selected_keywords)
    if include_monetary_info and 'selected_comp_keywords' in st.session_state:
        final_selected_keywords.extend(st.session_state.selected_comp_keywords)
    
    # Removing potential duplicates from selected keywords
    final_selected_keywords = list(set(final_selected_keywords))
    
    # Add to session_state
    st.session_state.translated_trans_keywords = translated_trans_keywords
    st.session_state.final_selected_keywords = final_selected_keywords

    st.write(translated_trans_keywords)

    # Separator
    st.markdown("---")
    
    
        
def research():
    st.title("Research 📚")

    # Ensure that the necessary data is in the session state
    if 'selected_keywords' not in st.session_state:
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
    if 'selected_keywords' not in st.session_state or 'translated_trans_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return

    # Load the CSV
    df = pd.read_csv(st.session_state.filename, encoding='utf-8')

    # Display the number of links in the sidebar
    st.sidebar.write(f"Total Links: {len(df)}")

    # Create an empty dataframe for sentence-level results
    sentence_df = pd.DataFrame(columns=['title', 'link', 'sentence_id', 'sentence'] + st.session_state.final_selected_keywords)

    # Check for 'Run Analysis' button click
    if st.sidebar.button("Run Analysis"):
        # For each keyword, create a new column initialized to 0
        for keyword in st.session_state.final_selected_keywords:
            df[keyword] = 0
        
        # Iterate through each link in the dataframe
        for index, row in df.iterrows():
            try:
                response = requests.get(row['link'])
                response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
                
                # Use BeautifulSoup to parse the page content
                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.get_text().lower()

                # Document-level keyword counting based on translated keywords
                for keyword, trans_keyword in zip(st.session_state.final_selected_keywords, st.session_state.translated_trans_keywords):
                    df.at[index, keyword] = text_content.count(trans_keyword.lower())

                # Sentence-level keyword counting based on translated keywords
                # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text_content)
                sentences = re.split(r'\n\s*\n', text_content)
                
                for sentence_id, sentence in enumerate(sentences, 1):
                    sentence_data = {'title': row['title'], 'link': row['link'], 'sentence_id': f"{index + 1}_{sentence_id}", 'sentence': sentence}
                    for keyword, trans_keyword in zip(st.session_state.final_selected_keywords, st.session_state.translated_trans_keywords):
                        sentence_data[keyword] = sentence.count(trans_keyword.lower())
                    sentence_df = sentence_df.append(sentence_data, ignore_index=True)
                    sentence_df['sentence'] = sentence_df['sentence'].str.replace('\n', ' ')
            
            except requests.RequestException:
                st.write(f"Error accessing {row['link']}")

        # Export the dataframes to CSVs
        df.to_csv('results/analyzed_results.csv', index=False, encoding='utf-8')
        sentence_df.to_csv('results/analyzed_results_sentences.csv', index=False, encoding='utf-8')
        
        st.write("Data exported to 'results/analyzed_results.csv'")
        st.write("Sentence data exported to 'results/analyzed_results_sentences.csv'")

def document_results():
    st.title("📚 Document-level Results")

    # Ensure that the necessary data is in the session state
    if 'selected_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return
    
    # Load the analyzed CSV
    df = pd.read_csv('results/analyzed_results.csv', encoding='utf-8')

    # Convert titles to hyperlinked titles
    df['title'] = df.apply(lambda row: f'<a href="{row["link"]}">{row["title"]}</a>', axis=1)

    # Drop the first column (assuming it's an index column) and 'link' column for displaying
    df_display = df.drop(columns=['link'])

    # Multiselect box with all the keywords as default
    keywords = st.session_state.final_selected_keywords
    selected_keywords = st.sidebar.multiselect("Select keywords:", options=keywords, default=keywords)

    # Ranking system based on selected keywords
    if len(selected_keywords) == 1:
        df_display = df_display.sort_values(by=selected_keywords[0], ascending=False)
    else:
        # Create a temporary dataframe to hold normalized values
        df_normalized = df_display[selected_keywords].copy()
        
        # Normalize the columns of selected keywords in the temporary dataframe
        for keyword in selected_keywords:
            df_normalized[keyword] = (df_normalized[keyword] - df_normalized[keyword].min()) / (df_normalized[keyword].max() - df_normalized[keyword].min())
        
        # Compute the average for each row across normalized columns
        df_display['normalized_avg'] = df_normalized.mean(axis=1)
        
        # Sort by the ranking column
        df_display = df_display.sort_values(by='normalized_avg', ascending=False)
        df_display = df_display.drop(columns=['normalized_avg'])
        
    # Display the modified dataframe
    st.write(df_display.to_html(escape=False), unsafe_allow_html=True)

def sentence_results():
    st.title("📚 Sentence-level Results")
    
    # Ensure that the necessary data is in the session state
    if 'selected_keywords' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return
    
    # Load the analyzed CSV for sentences
    df = pd.read_csv('results/analyzed_results_sentences.csv', encoding='utf-8')

    # Remove newline characters from the 'sentence' column
    df['sentence'] = df['sentence'].str.replace('\n', ' ')

    # Truncate lengthy sentences to a certain number of characters for display
    max_display_length = 10000000  # you can adjust this value
    df['sentence_display'] = df['sentence'].apply(lambda x: (x[:max_display_length] + '...') if isinstance(x, str) and len(x) > max_display_length else x)

    # Convert sentence IDs to hyperlinked IDs using the link column
    # df['sentence_id'] = df.apply(lambda row: f'<a href="{row["link"]}">{row["sentence_id"]}</a>', axis=1)
    df['sentence_id'] = df['sentence_id'].astype(str)

    # Drop unnecessary columns for displaying and use 'sentence_display' instead of 'sentence'
    df_display = df.drop(columns=['title', 'link', 'sentence'])
    
    # Multiselect box with all the keywords as default
    keywords = st.session_state.final_selected_keywords
    selected_keywords = st.sidebar.multiselect("Select keywords:", options=keywords, default=keywords)
    
    # Create a normalized_avg column if more than one keyword is selected
    if len(selected_keywords) > 1:
        df_normalized = df_display[selected_keywords].copy()
            
        for keyword in selected_keywords:
            df_normalized[keyword] = (df_normalized[keyword] - df_normalized[keyword].min()) / (df_normalized[keyword].max() - df_normalized[keyword].min())
            
        df_display['normalized_avg'] = df_normalized.mean(axis=1)
        histogram_column = 'normalized_avg'
    else:
        # If only one keyword, the histogram column will be that keyword
        histogram_column = selected_keywords[0]
           
    # For top X% Slider, choose the right column
    total_sentences = len(df_display)
    percentage = st.sidebar.slider(f"Select top X% based on {histogram_column}:", 0, 100, 5)
    top_x = int((percentage / 100) * total_sentences)
    
    # Create a matplotlib figure with desired size
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_display[histogram_column], bins=30, alpha=0.75)
    ax.set_title(f"Histogram of {histogram_column}")
    st.sidebar.pyplot(fig)
        
    # Display only the top X% of sentences based on the user's selection
    df_display = df_display.sort_values(by=histogram_column, ascending=False).head(top_x)
    
    # If normalized_avg column exists, drop it before displaying
    if 'normalized_avg' in df_display.columns:
        df_display = df_display.drop(columns=['normalized_avg'])
        
    # Display the dataframe
    st.dataframe(df_display)

def text_processing():
    st.title("📝 Text Processing")
    
    # Ensure that the necessary data is in the session state
    if 'total_characters' not in st.session_state:
        st.warning("Please complete the previous steps first.")
        return
    
    # Sidebar for model selection
    models = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4"]
    selected_model = st.sidebar.selectbox("Select OpenAI Model:", models, index=0, key="model_select_key")

    # Calculate tokens
    total_characters = st.session_state.total_characters
    tokens_estimate = total_characters / 4  # A rough estimate

    # Cost estimation based on the selected model
    if selected_model == "gpt-4":
        cost_per_token = 0.03 if tokens_estimate <= 8000 else 0.06
    elif selected_model in ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]:
        cost_per_token = 0.0015 if tokens_estimate <= 4000 else 0.003
    else:
        cost_per_token = 0

    estimated_cost = cost_per_token * (tokens_estimate/1000)
    
    st.sidebar.write(f"Estimated Tokens: {int(tokens_estimate)}")
    st.sidebar.write(f"Estimated Cost: ${estimated_cost:.5f}")

    # Information extraction preferences
    info_extraction_options = [
        "Summarize",
        "Investment info",
        "Projects info.",
        "Enter a custom extraction request."
    ]
    selected_extraction = st.sidebar.selectbox("Desired Action:", info_extraction_options, key="extraction_select_key")
    
    custom_request = ""
    if selected_extraction == "Enter a custom extraction request.":
        custom_request = st.sidebar.text_area("Specify your custom extraction request:", key="custom_request_key")

    # Display previously processed results if they exist
    if "processed_results" in st.session_state:
        for item in st.session_state.processed_results:
            st.subheader(f"[{item['original']['title']}]({item['original']['link']})")  # Displaying the original title as a clickable link
            st.write(f"Source: {item['original']['displayLink']}")
            st.write(f"Published Date: {item['original'].get('pagemap', {}).get('metatags', [{}])[0].get('og:updated_time', 'N/A')}")
            st.write(f"Processed Content: {item['processed']}")
            st.markdown("---")  # separator


    # When the process button is clicked
    if st.sidebar.button("Process Text"):
        # If results don't exist in the session state, initialize an empty list
        if "processed_results" not in st.session_state:
            st.session_state.processed_results = []

        # Clear previous results
        st.session_state.processed_results.clear()

        # Iterate over each result and process
        for result in st.session_state.results:
            snippet = result['snippet']
            
            # Depending on the selected extraction, modify the prompt
            if selected_extraction == info_extraction_options[0]:  # summary
                prompt = f"Provide a summary in 3 extended and comprehensive bullet points for the following text: {snippet}"
                prompt += f" . The resulting format MUST MUST MUST have first two bullets describing the document content. Then there bust a 3 bullet point list with key highlights. THIS IS THE FORMAT I WANT DON'T GIVE ME RESULTS IN ANY OTHER FORM. MAKE SURE YOU GO ON A NEW LINE AT EVERY BULLET POINT!!!!!!!!!!!!!!!"
            elif selected_extraction == info_extraction_options[1]:  # energy investments
                prompt = f"Identify and extract information related to energy investments from the following text: {snippet}"
            elif selected_extraction == info_extraction_options[2]:  # projects
                prompt = f"Identify and extract details of projects mentioned in the following text: {snippet}"
            else:  # custom
                prompt = custom_request
            
                        
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
                processed_text = response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    model=selected_model,
                    prompt=prompt,
                    max_tokens=200
                )
                processed_text = response.choices[0].text.strip()

            # Storing the processed result along with the original data
            st.session_state.processed_results.append({"original": result, "processed": processed_text})

        for item in st.session_state.processed_results:
            st.subheader(f"[{item['original']['title']}]({item['original']['link']})")   # Displaying the original title as a clickable link
            st.write(f"Source: {item['original']['displayLink']}")
            st.write(f"Published Date: {item['original'].get('pagemap', {}).get('metatags', [{}])[0].get('og:updated_time', 'N/A')}")
            st.write(f"Processed Content: {item['processed']}")
            st.markdown("---")   # separator


pages = {
    "🌍  Area Selection": area_selection,
    "✅ Selected Area Check ": selected_area_check,
    "🛠️ Define research": define_research,
    "🔍 Research": research,
    "🔍 Run Document Analysis": document_analysis,
    "📚 Document-level Results": document_results,
    "🔍 Text-level Results": sentence_results,
    "📝 Text Processing": text_processing,

}

selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
