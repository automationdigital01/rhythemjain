# Importing the libraries
import requests
from bs4 import BeautifulSoup
# import torch-----------------------------
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
# import streamlit as st
import pandas as pd
import urllib.parse
import urllib3
import multiprocessing
from goose3 import Goose
import streamlit as st

urllib3.disable_warnings()

# Function to convert search query to Google News search URL
def generate_google_news_url(query, num_links):
    encoded_query = urllib.parse.quote(query)
    
    # Generating list of google links, each containing 10 news links.
    google_links = []
    links_per_page= 10
    
    for page in range(0,num_links,links_per_page):
        google_news_url = f"https://www.google.com/search?q={encoded_query}&tbm=nws&num={min(num_links-page,10)}&start={page}"        
        google_links.append(google_news_url)
    return google_links


# Removing ads from text
def is_ad_element(tag):
    # To check if the tag has any of the ad-related classes
    ad_classes = ["ad", "advertisement", "sidebar", "popup"]
    return tag.get("class") and any(cls in tag.get("class") for cls in ad_classes)

# Function to extract the web links from google
def web_links(supplier, num_links):
    # Specify the search query with the company name
    search_query = f"{supplier} news"
    
    google_links = generate_google_news_url(search_query, num_links)
    
    links_list = []
    for google_link in google_links:
        data = requests.get(google_link)
        soup = BeautifulSoup(data.text, 'html.parser')
        
        # Iterate over the search results and extract news links
        for links in soup.find_all('a'):
            link = links.get('href')
            # Check if an unwanted link exists  
            if link and link.startswith("/url?q=") and filter_links(link):
                # Extract the actural URL from the Google search results
                actual_link = link.split("/url?q=")[1].split("&sa=")[0]
                links_list.append(actual_link)
    return links_list

#Extract text from the news links
def web_scrapping(url):
    try:
        headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}


        # Remove ad-related elements
        response = requests.get(url,verify=False, headers=headers)
        extractor = Goose()
        article = extractor.extract(raw_html = response.content)
        text= article.cleaned_text
        title = article.title
        #st.write(url)
        return text, title
    except:
        return "Unable to extract text", "Unable to extract title"
        #print("can't fetched the datafrom te url")

# parallel processing function
def summary_generation(text):
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)

    inputs = tokenizer.encode(
        "summarize: " + text, return_tensors="pt", max_length=512, truncation=True
    )
    summary_ids = model.generate(
        inputs, max_length=100, min_length=50, length_penalty=5.0, num_beams=2
    )
    summary = tokenizer.decode(summary_ids[0])
    summary = summary.replace("<pad>", "")
    summary = summary.replace("</s>", "")
    return summary

# get weblinks using news api
def weblink_news_api(company_name):
    # Replace 'YOUR_API_KEY' with your actual NewsAPI key
    api_key = "4e086fbfe2bc48eea914d5b05a79d498"
    proxy = None  # Set to None if you don't want to use a proxy

    try:
        # Create the URL for the NewsAPI request
        url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={api_key}&pageSize=10"

        # Send a GET request to NewsAPI with SSL verification disabled
        response = requests.get(url, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])

            # Create a set to store the unique websites that mention the company name
            company_websites = set()

            # Iterate through the articles and extract the source URLs
            for article in articles:
                source_url = article.get("url")

                # Check if the source URL is not None and not already in the set
                if source_url and source_url not in company_websites:
                    company_websites.add(source_url)

            return list(company_websites)
        else:
            print(
                f"Error: Unable to fetch news for {company_name}. Status code: {response.status_code}"
            )
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error: An error occurred during the request: {str(e)}")
        return None

# Function to filter out unwanted links
def filter_links(link):
    unwanted_domains = ["support.google.com", "accounts.google.com", "benzinga.com"]
    for domain in unwanted_domains:
        if domain in link:
            return False
    return True

finbert = BertForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone", num_labels=3
)
tokenizer_sentiment = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer_sentiment)

def multi_processing():
    df = pd.read_csv("C:\Work\Supplier Management\Company Names.csv")
    company_names = df['Entity Name'].to_list()
    
    pool = multiprocessing.Pool(processes=4)
    answers = pool.map(extract_summary_sentiment, company_names)
    final_result = []
    for ans in answers:
        final_result.extend(ans)
    

    # Close the pool
    pool.close()
    pool.join()

    return final_result
def extract_summary_sentiment1(company):
    links_list = web_links(company, 2)
    if links_list == None: 
        links_list = weblink_news_api(company)
    
    data = []
    print("Length of List: ", len(links_list))
    for link in links_list:
        ans = single_news(company,link)
        if ans['Title']!= "Unable to extract title":
            ans['Company'] = company
            data.append(ans)
            st.write("Title: " + ans['Title'])
            st.write("Summary: " + ans['Summary'])
            st.write("Analysis: " + ans['Sentiment'])
    return data
# def extract_summary_sentiment(company):
    
#     links_list = web_links(company, 13)
#     if links_list == None: 
#         links_list = weblink_news_api(company)

#     data = []
#     print("Length of List: ", len(links_list))
#     pool = multiprocessing.Pool(processes = 4)
#     answers = pool.map(single_news,links_list)
#     for ans in answers:
#         if ans['Title']!= "Unable to extract title":
#             ans['Company'] = company
#             data.extend(ans)
#             st.write("Title: " + ans['Title'])
#             st.write("Summary: " + ans['Summary'])
#             st.write("Analysis: " + ans['Sentiment'])
#     return data
 
def single_news(company,link):
    ans = dict()
    text,title = web_scrapping(link)
    if company not in title:
        ans['Title'] = "Unable to extract title"
        return ans
    summary = summary_generation(text)
    sentiment = nlp(summary)[0]["label"]
    ans['Sentiment'] = sentiment
    ans['Summary'] = summary
    ans['Title'] = title
    
    return ans
def main():
    """Streamlit User Interface"""
    header_container = st.container()

    with header_container:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.write("")
        with col2:
            st.write("")
            st.markdown(
                "<span style='color: #1E90FF'>Supplier's News Analysis</span> "
                " <span style='color: #92D293'></span>",
                unsafe_allow_html=True,
            )
        with col3:
            st.image(
                "C:\\Work\\Supplier Management\\transformers\\logo\\USER LOGin.png",
                width=70,
            )
            st.markdown(
                "<span style='color: #1E90FF'>Welcome User !</span>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <style>
            [data-testid=stSidebar] {
                background: linear-gradient(to bottom, #1E90FF, #92D293);
            }
        </style>
    """,
        unsafe_allow_html=True,
    )
    
    # Streamlit Sidebar
    st.sidebar.image(
        "C:\\Work\\Supplier Management\\transformers\\logo\\TECHNIP_ENERGIES_LOGO.png",
        width=100,
    )

    # st.title("Credit Analysis of Vendors")
    options = st.sidebar.multiselect(
        "Select the Suppliers",
        [
            "Halliburton Company",
            "Sick AG",
            "Sofinter S.p.a",
            "Chiyoda Corporation",
            "Atlas Copco Group",
            "Burckhardt Compression",
            "BALFOUR BEATTY PLC",
        ],
    )

    # List of URLs to block

    # #summarization using long-T5 summarizer, using huggingface
    # summarizer = pipeline("summarization", "pszemraj/long-t5-tglobal-base-16384-book-summary")

    # sentiment analysis using FinBert
    finbert = BertForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone", num_labels=3
    )
    tokenizer_sentiment = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer_sentiment)

    if st.sidebar.button("Submit"):
        st.write("Selected Suppliers:", options[0])
        result = extract_summary_sentiment1(options[0])
        df = pd.DataFrame(result)
        st.dataframe(df)
        csv = df.to_csv().encode("utf-8")

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="supplier_df.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
