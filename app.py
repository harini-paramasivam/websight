import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse, urljoin
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import time
import base64
from io import BytesIO
import random
from datetime import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
import difflib

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="WebSight - Advanced Web Scraper",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A4FEB;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9A9A9A;
        text-align: center;
        margin-top: 0;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #4A4FEB 0%, #7E84F3 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f0f0;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A4FEB;
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #4A4FEB;
        color:white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    div.stButton > button:hover {
        background-color: #2E32B8;
        color:white;
        border-color: #2E32B8;
    }
    .url-input {
        border-radius: 5px;
        border: 2px solid #4A4FEB;
    }
    .stProgress > div > div > div > div {
        background-color: #4A4FEB;
    }
    .highlight-box {
        background-color: #F0F2FF;
        border-left: 5px solid #4A4FEB;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .history-item {
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .history-item:hover {
        background-color: #f0f0f0;
    }
    .entity-box {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .entity-PERSON {
        background-color: #ffcdd2;
        border: 1px solid #e57373;
    }
    .entity-ORGANIZATION {
        background-color: #bbdefb;
        border: 1px solid #64b5f6;
    }
    .entity-LOCATION {
        background-color: #c8e6c9;
        border: 1px solid #81c784;
    }
    .entity-DATE {
        background-color: #fff9c4;
        border: 1px solid #fff176;
    }
    .entity-TIME {
        background-color: #ffe0b2;
        border: 1px solid #ffb74d;
    }
    .comparison-table th {
        background-color: #f0f2ff;
        padding: 8px;
    }
    .comparison-table td {
        text-align: center;
        padding: 8px;
    }
    .comparison-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Title and description with enhanced styling
st.markdown('<h1 class="main-header">🔍 WebSight</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Web Scraping & Content Analysis Tool</p>', unsafe_allow_html=True)

# Create sample data directory
if not os.path.exists('scraped_data'):
    os.makedirs('scraped_data')

# Initialize session states
if 'history' not in st.session_state:
    if os.path.exists('scraped_data/history.json'):
        try:
            with open('scraped_data/history.json', 'r') as f:
                st.session_state.history = json.load(f)
        except:
            st.session_state.history = []
    else:
        st.session_state.history = []

if 'scrape_count' not in st.session_state:
    st.session_state.scrape_count = 0

if 'comparison_urls' not in st.session_state:
    st.session_state.comparison_urls = []

# Function to validate URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Function to extract content using BeautifulSoup
def extract_content(url):
    try:
        # Send request with a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
            
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.text.strip()
            if href and text and not href.startswith('#') and not href.startswith('javascript:'):
                links.append({
                    'text': text,
                    'url': href
                })
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            alt = img.get('alt', '')
            if src:
                images.append({
                    'src': src,
                    'alt': alt
                })
        
        # Extract meta description
        meta_description = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_description = meta_tag.get('content', '')
        
        # Extract headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            headings.append({
                'type': h.name,
                'text': h.text.strip()
            })
        
        return {
            'title': title,
            'meta_description': meta_description,
            'text': text,
            'links': links,
            'images': images,
            'headings': headings,
            'html': str(soup)
        }
    except Exception as e:
        return {
            'error': str(e)
        }

# Function to extract main text content in a cleaner way
def extract_main_text(soup_html):
    soup = BeautifulSoup(soup_html, 'html.parser')
    
    # Try to find the main content area
    main_content = None
    
    # Common content containers
    content_candidates = soup.select('article, .content, .post, .entry, #content, #main, .main, .post-content, .entry-content')
    
    if content_candidates:
        # Select the candidate with the most text
        main_content = max(content_candidates, key=lambda x: len(x.get_text()))
    else:
        # Fallback to body if no candidate found
        main_content = soup.body
    
    if main_content:
        # Remove non-content elements
        for element in main_content.select('aside, nav, footer, header, .comments, .sidebar, .navigation, .menu, .ads, .advertisement'):
            element.extract()
            
        # Get cleaned text
        text = main_content.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        return clean_text
    
    return "Could not extract main content."

# Function to create a word cloud image
def generate_wordcloud(text):
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=150,
        stopwords=stop_words,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    # Convert to image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to base64 for display
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode()

# Function to extract named entities
def extract_entities(text):
    sentences = sent_tokenize(text[:10000])  # Limit to first 10k chars for performance
    entities = {
        'PERSON': [],
        'ORGANIZATION': [],
        'LOCATION': [],
        'GPE': [],  # Geo-Political Entity
        'FACILITY': [],
        'DATE': [],
        'TIME': [],
        'MONEY': [],
        'PERCENT': []
    }
    
    all_entities = []
    
    for sentence in sentences:
        try:
            tokens = nltk.word_tokenize(sentence)
            tagged = pos_tag(tokens)
            chunks = ne_chunk(tagged)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_type = chunk.label()
                    entity_value = ' '.join(c[0] for c in chunk)
                    if entity_type in entities:
                        entities[entity_type].append(entity_value)
                        all_entities.append((entity_value, entity_type))
        except Exception as e:
            continue
    
    # Remove duplicates while preserving order
    for entity_type in entities:
        entities[entity_type] = list(dict.fromkeys(entities[entity_type]))
    
    return entities, all_entities

# Function to scrape multiple pages from a site
def scrape_site_pages(base_url, max_pages=3):
    # Get the main page
    result = extract_content(base_url)
    if 'error' in result:
        return {'error': result['error']}
    
    # Get internal links on the same domain
    domain = urlparse(base_url).netloc
    internal_links = []
    for link in result['links']:
        link_url = link['url']
        # Convert relative URLs to absolute
        if not link_url.startswith(('http://', 'https://')):
            if link_url.startswith('/'):
                link_url = f"https://{domain}{link_url}"
            else:
                link_url = f"https://{domain}/{link_url}"
        
        # Check if it's on the same domain
        if urlparse(link_url).netloc == domain:
            internal_links.append(link_url)
    
    # Limit the number of pages to scrape
    internal_links = list(set(internal_links))[:max_pages]
    
    # Scrape each page in parallel
    all_results = {'main_page': result, 'other_pages': {}}
    
    def scrape_page(url):
        try:
            page_result = extract_content(url)
            return url, page_result
        except Exception as e:
            return url, {'error': str(e)}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(scrape_page, url) for url in internal_links]
        for future in futures:
            url, page_result = future.result()
            all_results['other_pages'][url] = page_result
    
    return all_results

# Function to compare content across similar websites
def compare_websites(urls, analyze_content=True):
    results = {}
    analyses = {}
    
    # Scrape each website
    for url in urls:
        results[url] = extract_content(url)
        if 'error' not in results[url] and analyze_content:
            main_text = extract_main_text(results[url]['html'])
            analyses[url] = analyze_text(main_text)
    
    # Compare titles, descriptions, headings
    comparison = {
        'titles': {},
        'descriptions': {},
        'word_counts': {},
        'common_words_overlap': {},
        'sentiment': {},
        'readability': {}
    }
    
    # Get all titles
    for url, result in results.items():
        if 'error' not in result:
            comparison['titles'][url] = result['title']
            comparison['descriptions'][url] = result['meta_description']
            
            if analyze_content and url in analyses:
                comparison['word_counts'][url] = analyses[url]['word_count']
                comparison['sentiment'][url] = analyses[url]['sentiment']['category']
                comparison['readability'][url] = analyses[url]['readability']['level']
    
    # Compare common words across sites
    if analyze_content and len(analyses) > 1:
        all_common_words = {}
        for url, analysis in analyses.items():
            all_common_words[url] = dict(analysis['common_words'])
        
        # Find overlapping words
        all_words = set()
        for url, words in all_common_words.items():
            all_words.update(words.keys())
        
        overlap = {}
        for word in all_words:
            overlap[word] = []
            for url, words in all_common_words.items():
                if word in words:
                    overlap[word].append((url, words[word]))
        
        # Keep only words that appear on multiple sites
        comparison['common_words_overlap'] = {
            word: sites for word, sites in overlap.items() 
            if len(sites) > 1
        }
    
    return comparison

# Function to analyze text with advanced features
def analyze_text(text):
    # Basic statistics
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    char_count = len(text)
    line_count = text.count('\n') + 1
    
    # Get stop words
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback to simple stop words if NLTK data isn't available
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 
                     'when', 'at', 'from', 'by', 'for', 'with', 'about', 'to'}
    
    # Filter stop words and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Calculate word frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
    
    # Estimated reading time (250 words per minute is average reading speed)
    reading_time = round(word_count / 250) if word_count > 0 else 0
    
    # Sentiment analysis using a simple approach
    positive_words = {'good', 'great', 'excellent', 'best', 'amazing', 'awesome',
                     'wonderful', 'fantastic', 'terrific', 'outstanding', 'superb',
                     'brilliant', 'perfect', 'happy', 'pleased', 'delighted', 'love',
                     'enjoy', 'beneficial', 'positive', 'impressive', 'recommended'}
    
    negative_words = {'bad', 'worst', 'terrible', 'awful', 'horrible', 'poor',
                     'disappointing', 'negative', 'difficult', 'hard', 'problem',
                     'issue', 'concern', 'hate', 'dislike', 'unfortunately',
                     'fail', 'failure', 'inadequate', 'useless', 'unpleasant'}
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total_sentiment_words = positive_count + negative_count
    
    # Calculate sentiment score (range -1 to 1)
    sentiment_score = 0
    if total_sentiment_words > 0:
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
    
    # Determine sentiment category
    if sentiment_score > 0.2:
        sentiment = "Positive"
    elif sentiment_score < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Readability score (simple approximation of Flesch Reading Ease)
    # Count sentences approximately
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences == 0:  # Avoid division by zero
        sentences = 1
    
    # Average words per sentence
    avg_words_per_sentence = word_count / sentences
    
    # Average characters per word
    avg_chars_per_word = char_count / word_count if word_count > 0 else 0
    
    # Simple readability calculation (higher is easier to read)
    readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_chars_per_word)
    readability = min(max(readability, 0), 100)  # Clamp to 0-100 range
    
    # Determining readability level
    if readability > 90:
        readability_level = "Very Easy"
    elif readability > 80:
        readability_level = "Easy"
    elif readability > 70:
        readability_level = "Fairly Easy"
    elif readability > 60:
        readability_level = "Standard"
    elif readability > 50:
        readability_level = "Fairly Difficult"
    elif readability > 30:
        readability_level = "Difficult"
    else:
        readability_level = "Very Difficult"
    
    # Generate WordCloud
    wordcloud_base64 = None
    if word_count > 30:  # Only create word cloud if there's enough text
        try:
            wordcloud_base64 = generate_wordcloud(text)
        except Exception as e:
            wordcloud_base64 = None
    
    # Get sentence distribution by length (number of words)
    sentences_by_length = {}
    for sentence in re.split(r'[.!?]', text):
        sentence = sentence.strip()
        if sentence:
            word_count_in_sentence = len(re.findall(r'\b\w+\b', sentence))
            if word_count_in_sentence in sentences_by_length:
                sentences_by_length[word_count_in_sentence] += 1
            else:
                sentences_by_length[word_count_in_sentence] = 1
    
    # Extract named entities
    try:
        entities, entity_list = extract_entities(text)
    except Exception as e:
        entities = {}
        entity_list = []
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'line_count': line_count,
        'common_words': common_words,
        'reading_time': reading_time,
        'sentiment': {
            'score': sentiment_score,
            'category': sentiment,
            'positive_words': positive_count,
            'negative_words': negative_count
        },
        'readability': {
            'score': readability,
            'level': readability_level,
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_chars_per_word': avg_chars_per_word
        },
        'wordcloud': wordcloud_base64,
        'sentences_by_length': sentences_by_length,
        'entities': entities,
        'entity_list': entity_list
    }

# Main content area with enhanced styling
st.markdown('<div class="feature-card">', unsafe_allow_html=True)
st.markdown("### Enter a website URL to analyze")
st.markdown("Paste the full URL including http:// or https:// to extract content and analyze it.")
st.markdown('</div>', unsafe_allow_html=True)

# App tabs for main functionality
main_tabs = st.tabs(["Single Website Analysis", "Multi-page Scraping", "Website Comparison"])

with main_tabs[0]:
    # Enhanced input form
    with st.form("url_form"):
        # More attractive URL input
        st.markdown('<style>.url-input .stTextInput input {border: 2px solid #4A4FEB; border-radius: 5px; padding: 10px;}</style>', unsafe_allow_html=True)
        url = st.text_input("Enter a website URL:", placeholder="https://example.com", key="url_input")
        
        # Form columns for options
        option_cols = st.columns(3)
        with option_cols[0]:
            extract_entities_option = st.checkbox("Extract named entities", value=True, 
                                    help="Extract people, organizations, locations, and other entities")
        with option_cols[1]:
            include_images = st.checkbox("Analyze images", value=True,
                                    help="Download and analyze images from the website")
        with option_cols[2]:
            save_results = st.checkbox("Auto-save results", value=False,
                                    help="Automatically save results to the history")
        
        # Submit button with styling
        submitted = st.form_submit_button("🔍 Scrape Website")

    if submitted:
        if is_valid_url(url):
            # Create a progress bar for better UX
            progress_bar = st.progress(0)
            
            # Add a status message
            status_message = st.empty()
            status_message.info("Connecting to website...")
            time.sleep(0.5)  # Small delay for visual effect
            
            # Update progress
            progress_bar.progress(25)
            status_message.info("Downloading content...")
            time.sleep(0.5)  # Small delay for visual effect
            
            with st.spinner("Analyzing content..."):
                result = extract_content(url)
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display the results
                    st.success("Website scraped successfully!")
                    
                    # Tabs for different sections
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Main Content", "Links", "Images", "Analysis", "Named Entities"])
                    
                    with tab1:
                        st.header("Website Overview")
                        st.subheader("Title")
                        st.write(result['title'])
                        
                        if result['meta_description']:
                            st.subheader("Meta Description")
                            st.write(result['meta_description'])
                        
                        st.subheader("Headings")
                        for heading in result['headings'][:10]:  # Show first 10 headings
                            st.write(f"**{heading['type'].upper()}:** {heading['text']}")
                    
                    with tab2:
                        st.header("Main Content")
                        main_text = extract_main_text(result['html'])
                        
                        # Show excerpt
                        st.subheader("Content Excerpt")
                        st.write(main_text[:1000] + "..." if len(main_text) > 1000 else main_text)
                        
                        # Download button for full text
                        st.download_button(
                            label="Download Full Text",
                            data=main_text,
                            file_name=f"content_{urlparse(url).netloc}.txt",
                            mime="text/plain"
                        )
                        
                        # Full text in expander
                        with st.expander("Show Full Text"):
                            st.text_area("", main_text, height=400)
                    
                    with tab3:
                        st.header("Links")
                        if result['links']:
                            links_df = pd.DataFrame(result['links'])
                            st.dataframe(links_df)
                            
                            # Download button for links
                            csv = links_df.to_csv(index=False)
                            st.download_button(
                                label="Download Links as CSV",
                                data=csv,
                                file_name=f"links_{urlparse(url).netloc}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.write("No links found.")
                    
                    with tab4:
                        st.header("Images")
                        if result['images']:
                            # Display image count
                            st.write(f"Found {len(result['images'])} images")
                            
                            # Create a dataframe for the images
                            images_df = pd.DataFrame(result['images'])
                            st.dataframe(images_df)
                            
                            # Show a few sample images
                            st.subheader("Sample Images")
                            cols = st.columns(3)
                            for i, img in enumerate(result['images'][:6]):  # Show first 6 images
                                try:
                                    if not img['src'].startswith(('http://', 'https://')):
                                        # Handle relative URLs
                                        base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
                                        img_url = base_url + img['src'] if img['src'].startswith('/') else base_url + '/' + img['src']
                                    else:
                                        img_url = img['src']
                                    
                                    cols[i % 3].image(img_url, caption=img['alt'][:30], width=150)
                                except Exception as e:
                                    cols[i % 3].write(f"Unable to load image: {str(e)}")
                        else:
                            st.write("No images found.")
                    
                    with tab5:
                        st.header("Advanced Text Analysis")
                        
                        # Analyze the main text
                        analysis = analyze_text(main_text)
                        
                        # Create columns for metrics
                        metrics_cols = st.columns(4)
                        
                        # Basic stats with better styling
                        with metrics_cols[0]:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Word Count", analysis['word_count'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with metrics_cols[1]:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Reading Time", f"{analysis['reading_time']} min")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with metrics_cols[2]:
                            sentiment = analysis['sentiment']['category']
                            sentiment_icon = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "☹️"
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Sentiment", f"{sentiment_icon} {sentiment}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with metrics_cols[3]:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Readability", analysis['readability']['level'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Word Cloud visualization
                        if analysis['wordcloud']:
                            st.subheader("Word Cloud")
                            st.markdown(f'<img src="data:image/png;base64,{analysis["wordcloud"]}" width="100%">', unsafe_allow_html=True)
                        
                        # Create tabs for detailed analysis
                        analysis_tabs = st.tabs(["Common Words", "Sentiment Analysis", "Readability", "Text Statistics"])
                        
                        with analysis_tabs[0]:
                            st.subheader("Most Common Words")
                            # Show common words
                            common_words_df = pd.DataFrame(analysis['common_words'][:20], columns=['Word', 'Frequency'])
                            
                            # Two column layout for chart and table
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                st.bar_chart(common_words_df.set_index('Word')['Frequency'])
                            
                            with col2:
                                st.dataframe(common_words_df, height=300)
                                
                            # Show download button for all words
                            if len(analysis['common_words']) > 20:
                                all_words_df = pd.DataFrame(analysis['common_words'], columns=['Word', 'Frequency'])
                                csv = all_words_df.to_csv(index=False)
                                st.download_button(
                                    label="Download All Words as CSV",
                                    data=csv,
                                    file_name=f"words_{urlparse(url).netloc}.csv",
                                    mime="text/csv"
                                )
                        
                        with analysis_tabs[1]:
                            st.subheader("Sentiment Analysis")
                            
                            # Sentiment score with gauge
                            score = analysis['sentiment']['score']
                            # Create a simple gauge visualization
                            fig, ax = plt.subplots(figsize=(10, 2))
                            ax.barh([0], [1], color='lightgray', height=0.3)
                            ax.barh([0], [0.5 + score/2], color='blue' if score >= 0 else 'red', height=0.3)
                            ax.set_xlim(0, 1)
                            ax.set_ylim(-0.5, 0.5)
                            ax.get_yaxis().set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            
                            # Add markers for negative, neutral, positive
                            ax.text(0.1, -0.3, 'Negative', ha='center')
                            ax.text(0.5, -0.3, 'Neutral', ha='center')
                            ax.text(0.9, -0.3, 'Positive', ha='center')
                            
                            # Add pointer for current score
                            pointer_pos = 0.5 + score/2
                            ax.annotate('', xy=(pointer_pos, 0), xytext=(pointer_pos, 0.2),
                                        arrowprops=dict(arrowstyle='wedge', color='black'))
                            
                            st.pyplot(fig)
                            
                            # Sentiment details
                            st.markdown(f"<div class='highlight-box'>", unsafe_allow_html=True)
                            st.markdown(f"**Sentiment Score:** {score:.2f} (-1 to +1 scale)")
                            st.markdown(f"**Detected Positive Words:** {analysis['sentiment']['positive_words']}")
                            st.markdown(f"**Detected Negative Words:** {analysis['sentiment']['negative_words']}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with analysis_tabs[2]:
                            st.subheader("Readability Analysis")
                            
                            # Readability score visualization
                            readability = analysis['readability']['score']
                            
                            # Create a readability gauge
                            fig, ax = plt.subplots(figsize=(10, 3))
                            
                            # Create gauge background with color gradient
                            cmap = plt.cm.RdYlGn  # Red to Yellow to Green colormap
                            norm = plt.Normalize(0, 100)
                            
                            # Draw the gauge background
                            for i in range(100):
                                ax.barh([0], [1], left=[i], color=[cmap(norm(i))], height=0.3, alpha=0.7)
                            
                            # Add the pointer
                            ax.annotate('', xy=(readability, 0), xytext=(readability, 0.2),
                                       arrowprops=dict(arrowstyle='wedge', color='black'))
                            
                            # Add readability level markers
                            levels = [
                                (0, "Very Difficult"), 
                                (30, "Difficult"),
                                (50, "Fairly Difficult"),
                                (60, "Standard"),
                                (70, "Fairly Easy"),
                                (80, "Easy"),
                                (90, "Very Easy")
                            ]
                            
                            for pos, label in levels:
                                ax.axvline(x=pos, color='black', alpha=0.3, linestyle='--')
                                ax.text(pos, -0.3, label, ha='center', fontsize=8, rotation=45)
                            
                            # Add current score text
                            ax.text(readability, 0.4, f"{readability:.1f}", ha='center', fontweight='bold')
                            
                            # Clean up the axes
                            ax.set_xlim(0, 100)
                            ax.set_ylim(-0.5, 0.5)
                            ax.get_yaxis().set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.set_title("Flesch Reading Ease Score (higher = easier to read)")
                            
                            st.pyplot(fig)
                            
                            # Readability details
                            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
                            st.markdown(f"**Readability Level:** {analysis['readability']['level']}")
                            st.markdown(f"**Avg. Words per Sentence:** {analysis['readability']['avg_words_per_sentence']:.1f}")
                            st.markdown(f"**Avg. Characters per Word:** {analysis['readability']['avg_chars_per_word']:.1f}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Sentence distribution visualization if available
                            if analysis['sentences_by_length']:
                                st.subheader("Sentence Length Distribution")
                                sentence_df = pd.DataFrame(
                                    [(length, count) for length, count in analysis['sentences_by_length'].items()],
                                    columns=['Words per Sentence', 'Count']
                                )
                                sentence_df = sentence_df.sort_values('Words per Sentence')
                                st.bar_chart(sentence_df.set_index('Words per Sentence')['Count'])
                        
                        with analysis_tabs[3]:
                            st.subheader("Text Statistics")
                            
                            # Additional statistics
                            stats_cols = st.columns(2)
                            
                            with stats_cols[0]:
                                st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
                                st.markdown("### Basic Statistics")
                                st.markdown(f"**Total Characters:** {analysis['char_count']}")
                                st.markdown(f"**Total Words:** {analysis['word_count']}")
                                st.markdown(f"**Total Lines:** {analysis['line_count']}")
                                st.markdown(f"**Estimated Reading Time:** {analysis['reading_time']} minutes")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with stats_cols[1]:
                                # Calculate paragraph stats
                                paragraphs = [p for p in main_text.split('\n\n') if p.strip()]
                                avg_paragraph_length = sum(len(re.findall(r'\b\w+\b', p)) for p in paragraphs) / len(paragraphs) if paragraphs else 0
                                
                                st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
                                st.markdown("### Structure Statistics")
                                st.markdown(f"**Paragraphs:** {len(paragraphs)}")
                                st.markdown(f"**Avg. Words per Paragraph:** {avg_paragraph_length:.1f}")
                                st.markdown(f"**Headings:** {len(result['headings'])}")
                                st.markdown(f"**Links:** {len(result['links'])}")
                                st.markdown(f"**Images:** {len(result['images'])}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Save analysis data
                            if st.button("Save Analysis Data"):
                                # Create a timestamp for the filename
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"scraped_data/analysis_{urlparse(url).netloc}_{timestamp}.json"
                                
                                # Prepare data for saving (remove wordcloud as it's too large)
                                save_data = {**analysis}
                                save_data.pop('wordcloud', None)
                                
                                # Add metadata
                                save_data['metadata'] = {
                                    'url': url,
                                    'title': result['title'],
                                    'timestamp': timestamp
                                }
                                
                                # Save to file
                                with open(filename, 'w') as f:
                                    json.dump(save_data, f)
                                
                                # Add to history
                                history_entry = {
                                    'url': url,
                                    'title': result['title'],
                                    'timestamp': timestamp,
                                    'word_count': analysis['word_count'],
                                    'sentiment': analysis['sentiment']['category'],
                                    'filename': filename
                                }
                                
                                st.session_state.history.append(history_entry)
                                
                                # Save history
                                with open('scraped_data/history.json', 'w') as f:
                                    json.dump(st.session_state.history, f)
                                
                                st.success(f"Analysis data saved: {filename}")
                    
                    with tab6:
                        st.header("Named Entities")
                        
                        # Display only if entity extraction was enabled and entities were found
                        if extract_entities_option and 'entities' in analysis and analysis['entities']:
                            # Create columns for different entity types
                            entity_cols = st.columns(3)
                            
                            # Helper function to display entity list with styling
                            def display_entities(entity_type, title, column, color_class):
                                if entity_type in analysis['entities'] and analysis['entities'][entity_type]:
                                    column.subheader(title)
                                    for entity in analysis['entities'][entity_type]:
                                        column.markdown(f"<span class='entity-box entity-{color_class}'>{entity}</span>", unsafe_allow_html=True)
                            
                            # Display entities by type
                            display_entities('PERSON', '👤 People', entity_cols[0], 'PERSON')
                            display_entities('ORGANIZATION', '🏢 Organizations', entity_cols[1], 'ORGANIZATION')
                            display_entities('LOCATION', '📍 Locations', entity_cols[1], 'LOCATION')
                            display_entities('GPE', '🗺️ Places', entity_cols[1], 'LOCATION')
                            display_entities('DATE', '📅 Dates', entity_cols[2], 'DATE')
                            display_entities('TIME', '⏰ Times', entity_cols[2], 'TIME')
                            display_entities('MONEY', '💰 Money', entity_cols[2], 'DATE')
                            display_entities('PERCENT', '📊 Percentages', entity_cols[2], 'DATE')
                            
                            # Show text with highlighted entities
                            st.subheader("Text with Highlighted Entities")
                            
                            if analysis['entity_list']:
                                # Create a version of the text with highlighted entities
                                highlighted_text = main_text[:5000] # First 5000 chars for performance
                                
                                # Apply highlighting to all entities (primitive approach)
                                entity_spans = []
                                for entity, entity_type in analysis['entity_list']:
                                    entity_lower = entity.lower()
                                    text_lower = highlighted_text.lower()
                                    start_idx = 0
                                    while True:
                                        idx = text_lower.find(entity_lower, start_idx)
                                        if idx == -1:
                                            break
                                        entity_spans.append((idx, idx + len(entity), entity, entity_type))
                                        start_idx = idx + 1
                                
                                # Sort spans by start index
                                entity_spans.sort(key=lambda x: x[0])
                                
                                # Merge overlapping spans
                                merged_spans = []
                                for span in entity_spans:
                                    if not merged_spans or span[0] >= merged_spans[-1][1]:
                                        merged_spans.append(span)
                                
                                # Build highlighted html
                                html_parts = []
                                last_end = 0
                                for start, end, entity, entity_type in merged_spans:
                                    if start > last_end:
                                        html_parts.append(highlighted_text[last_end:start])
                                    html_parts.append(f'<span class="entity-box entity-{entity_type}">{highlighted_text[start:end]}</span>')
                                    last_end = end
                                
                                if last_end < len(highlighted_text):
                                    html_parts.append(highlighted_text[last_end:])
                                
                                highlighted_html = ''.join(html_parts)
                                
                                # Display in an expander
                                with st.expander("Show Text with Highlighted Entities"):
                                    st.markdown(highlighted_html, unsafe_allow_html=True)
                            
                            # Download button for entities
                            entity_data = {entity_type: entities for entity_type, entities in analysis['entities'].items() if entities}
                            st.download_button(
                                label="Download Entities as JSON",
                                data=json.dumps(entity_data, indent=2),
                                file_name=f"entities_{urlparse(url).netloc}.json",
                                mime="application/json"
                            )
                        else:
                            st.info("No named entities were found or entity extraction was disabled.")

                    # Automatically save results if the option was selected
                    if save_results:
                        # Create a timestamp for the filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"scraped_data/analysis_{urlparse(url).netloc}_{timestamp}.json"
                        
                        # Prepare data for saving (remove wordcloud as it's too large)
                        save_data = {**analysis}
                        if 'wordcloud' in save_data:
                            save_data.pop('wordcloud', None)
                        
                        # Add metadata
                        save_data['metadata'] = {
                            'url': url,
                            'title': result['title'],
                            'timestamp': timestamp
                        }
                        
                        # Save to file
                        with open(filename, 'w') as f:
                            json.dump(save_data, f)
                        
                        # Add to history
                        history_entry = {
                            'url': url,
                            'title': result['title'],
                            'timestamp': timestamp,
                            'word_count': analysis['word_count'],
                            'sentiment': analysis['sentiment']['category'],
                            'filename': filename
                        }
                        
                        st.session_state.history.append(history_entry)
                        
                        # Save history
                        with open('scraped_data/history.json', 'w') as f:
                            json.dump(st.session_state.history, f)
                        
                        st.success(f"Analysis data automatically saved to: {filename}")
        else:
            st.error("Please enter a valid URL (include 'http://' or 'https://')")

with main_tabs[1]:
    st.header("Multi-page Website Scraping")
    st.markdown("""
    This tool allows you to scrape multiple pages from the same website at once.
    Enter the main URL of the website, and the tool will automatically detect and follow internal links.
    """)
    
    with st.form("multipage_form"):
        multi_url = st.text_input("Enter the main website URL:", placeholder="https://example.com", key="multi_url_input")
        max_pages = st.slider("Maximum number of pages to scrape:", min_value=1, max_value=10, value=3)
        multi_submitted = st.form_submit_button("🔍 Scrape Multiple Pages")
    
    if multi_submitted:
        if is_valid_url(multi_url):
            with st.spinner(f"Scraping up to {max_pages} pages from {multi_url}..."):
                results = scrape_site_pages(multi_url, max_pages)
                
                if 'error' in results:
                    st.error(f"Error: {results['error']}")
                else:
                    # Display main page info
                    st.success(f"Successfully scraped the main page and {len(results['other_pages'])} additional pages!")
                    
                    # Create tabs for each page
                    main_page = results['main_page']
                    other_pages = results['other_pages']
                    
                    # Show summary table
                    st.subheader("Pages Scraped")
                    
                    # Prepare summary data
                    summary_data = []
                    
                    # Add main page
                    main_domain = urlparse(multi_url).netloc
                    main_title = main_page['title']
                    main_text = extract_main_text(main_page['html'])
                    main_word_count = len(re.findall(r'\b\w+\b', main_text))
                    summary_data.append({
                        'Page': 'Main Page',
                        'URL': multi_url,
                        'Title': main_title,
                        'Word Count': main_word_count,
                        'Links': len(main_page['links']),
                        'Images': len(main_page['images'])
                    })
                    
                    # Add other pages
                    for i, (page_url, page_data) in enumerate(other_pages.items()):
                        if 'error' not in page_data:
                            page_text = extract_main_text(page_data['html'])
                            page_word_count = len(re.findall(r'\b\w+\b', page_text))
                            summary_data.append({
                                'Page': f"Page {i+1}",
                                'URL': page_url,
                                'Title': page_data['title'],
                                'Word Count': page_word_count,
                                'Links': len(page_data['links']),
                                'Images': len(page_data['images'])
                            })
                    
                    # Display summary table
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)
                    
                    # Analyze content across all pages
                    if st.button("Analyze Content Across All Pages"):
                        with st.spinner("Analyzing content across all pages..."):
                            # Combine text from all pages
                            all_text = main_text
                            for page_url, page_data in other_pages.items():
                                if 'error' not in page_data:
                                    page_text = extract_main_text(page_data['html'])
                                    all_text += "\n\n" + page_text
                            
                            # Analyze combined text
                            combined_analysis = analyze_text(all_text)
                            
                            # Display results
                            st.subheader("Combined Content Analysis")
                            
                            # Basic stats
                            metrics_cols = st.columns(4)
                            with metrics_cols[0]:
                                st.metric("Total Words", combined_analysis['word_count'])
                            with metrics_cols[1]:
                                st.metric("Reading Time", f"{combined_analysis['reading_time']} min")
                            with metrics_cols[2]:
                                st.metric("Sentiment", combined_analysis['sentiment']['category'])
                            with metrics_cols[3]:
                                st.metric("Readability", combined_analysis['readability']['level'])
                            
                            # WordCloud for all content
                            if combined_analysis['wordcloud']:
                                st.subheader("Word Cloud (All Pages)")
                                st.markdown(f'<img src="data:image/png;base64,{combined_analysis["wordcloud"]}" width="100%">', unsafe_allow_html=True)
                            
                            # Show top words
                            st.subheader("Most Common Words Across All Pages")
                            common_words_df = pd.DataFrame(combined_analysis['common_words'][:30], columns=['Word', 'Frequency'])
                            st.bar_chart(common_words_df.set_index('Word')['Frequency'])
                            
                            # Show named entities if available
                            if extract_entities_option and 'entities' in combined_analysis:
                                st.subheader("Named Entities Across All Pages")
                                entity_counts = {}
                                for entity_type, entities in combined_analysis['entities'].items():
                                    entity_counts[entity_type] = len(entities)
                                
                                # Display entity counts
                                entity_df = pd.DataFrame([(k, v) for k, v in entity_counts.items() if v > 0], 
                                                        columns=['Entity Type', 'Count'])
                                st.bar_chart(entity_df.set_index('Entity Type')['Count'])
                    
                    # Option to download all data
                    if st.button("Download All Scraped Data"):
                        # Prepare data for download
                        download_data = {
                            'main_page': {
                                'url': multi_url,
                                'title': main_page['title'],
                                'meta_description': main_page['meta_description'],
                                'content': main_text,
                                'links': main_page['links'],
                                'headings': main_page['headings']
                            },
                            'other_pages': {}
                        }
                        
                        for page_url, page_data in other_pages.items():
                            if 'error' not in page_data:
                                page_text = extract_main_text(page_data['html'])
                                download_data['other_pages'][page_url] = {
                                    'title': page_data['title'],
                                    'meta_description': page_data['meta_description'],
                                    'content': page_text,
                                    'links': page_data['links'],
                                    'headings': page_data['headings']
                                }
                        
                        # Convert to JSON
                        json_data = json.dumps(download_data, indent=2)
                        domain = urlparse(multi_url).netloc
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Provide download button
                        st.download_button(
                            label="Download as JSON",
                            data=json_data,
                            file_name=f"website_{domain}_{timestamp}.json",
                            mime="application/json"
                        )
        else:
            st.error("Please enter a valid URL (include 'http://' or 'https://')")

with main_tabs[2]:
    st.header("Website Comparison")
    st.markdown("""
    This tool allows you to compare the content, structure, and SEO aspects of multiple websites.
    Enter up to 3 URLs to compare their content and gain insights into their similarities and differences.
    """)
    
    # Form for entering comparison URLs
    with st.form("comparison_form"):
        st.subheader("Enter URLs to Compare")
        url1 = st.text_input("URL 1:", placeholder="https://example1.com", key="url1_input")
        url2 = st.text_input("URL 2:", placeholder="https://example2.com", key="url2_input")
        url3 = st.text_input("URL 3 (optional):", placeholder="https://example3.com", key="url3_input")
        
        compare_submitted = st.form_submit_button("🔍 Compare Websites")
    
    if compare_submitted:
        # Collect valid URLs
        urls = [url for url in [url1, url2, url3] if url and is_valid_url(url)]
        
        if len(urls) < 2:
            st.error("Please enter at least 2 valid URLs for comparison.")
        else:
            with st.spinner(f"Comparing {len(urls)} websites..."):
                # Perform comparison
                comparison = compare_websites(urls)
                
                # Display results
                st.success(f"Successfully compared {len(urls)} websites!")
                
                # Display title and description comparison
                st.subheader("Basic Comparison")
                
                # Create comparison table
                comparison_data = []
                for url in urls:
                    domain = urlparse(url).netloc
                    row = {
                        'Website': domain,
                        'Title': comparison['titles'].get(url, "N/A"),
                        'Meta Description': comparison['descriptions'].get(url, "N/A"),
                        'Word Count': comparison['word_counts'].get(url, "N/A"),
                        'Sentiment': comparison['sentiment'].get(url, "N/A"),
                        'Readability': comparison['readability'].get(url, "N/A")
                    }
                    comparison_data.append(row)
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Common words overlap
                if 'common_words_overlap' in comparison and comparison['common_words_overlap']:
                    st.subheader("Common Words Across Websites")
                    
                    # Get top 20 overlapping words
                    top_overlaps = sorted(
                        comparison['common_words_overlap'].items(),
                        key=lambda x: sum(count for _, count in x[1]),
                        reverse=True
                    )[:20]
                    
                    # Create data for visualization
                    overlap_data = []
                    for word, occurrences in top_overlaps:
                        for url, count in occurrences:
                            domain = urlparse(url).netloc
                            overlap_data.append({
                                'Word': word,
                                'Website': domain,
                                'Frequency': count
                            })
                    
                    overlap_df = pd.DataFrame(overlap_data)
                    
                    # Create grouped bar chart
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Get unique words and websites
                    words = overlap_df['Word'].unique()
                    websites = overlap_df['Website'].unique()
                    
                    # Set up positions
                    x = np.arange(len(words))
                    width = 0.8 / len(websites)
                    
                    # Create bars for each website
                    for i, website in enumerate(websites):
                        website_data = overlap_df[overlap_df['Website'] == website]
                        frequencies = []
                        for word in words:
                            word_data = website_data[website_data['Word'] == word]
                            if not word_data.empty:
                                frequencies.append(word_data['Frequency'].values[0])
                            else:
                                frequencies.append(0)
                        
                        ax.bar(x + i * width - 0.4 + width/2, frequencies, width, label=website)
                    
                    # Set labels and title
                    ax.set_title('Common Words Across Websites')
                    ax.set_xlabel('Words')
                    ax.set_ylabel('Frequency')
                    ax.set_xticks(x)
                    ax.set_xticklabels(words, rotation=45, ha='right')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Create HTML table for detailed comparison
                    st.subheader("Detailed Word Comparison")
                    
                    html_table = """
                    <table class="comparison-table" style="width:100%">
                        <tr>
                            <th>Word</th>
                    """
                    
                    # Add website columns
                    for url in urls:
                        domain = urlparse(url).netloc
                        html_table += f"<th>{domain}</th>"
                    
                    html_table += "</tr>"
                    
                    # Add rows for each word
                    for word, occurrences in top_overlaps:
                        html_table += f"<tr><td><b>{word}</b></td>"
                        
                        # Dictionary to look up frequency by URL
                        freq_by_url = {url: count for url, count in occurrences}
                        
                        for url in urls:
                            freq = freq_by_url.get(url, 0)
                            cell_bg = f"background-color: rgba(74, 79, 235, {min(freq/50, 0.8)})" if freq > 0 else ""
                            html_table += f'<td style="{cell_bg}">{freq if freq > 0 else "-"}</td>'
                        
                        html_table += "</tr>"
                    
                    html_table += "</table>"
                    
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                # Add to comparison history
                if st.button("Save Comparison to History"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create comparison entry for history
                    comparison_entry = {
                        'timestamp': timestamp,
                        'urls': urls,
                        'domains': [urlparse(url).netloc for url in urls]
                    }
                    
                    # Initialize comparison list if needed
                    if 'comparisons' not in st.session_state:
                        st.session_state.comparisons = []
                    
                    # Add to history
                    st.session_state.comparisons.append(comparison_entry)
                    
                    # Save to file
                    with open('scraped_data/comparisons.json', 'w') as f:
                        json.dump(st.session_state.comparisons, f)
                    
                    st.success("Comparison saved to history!")
                    
                # Export comparison data
                export_data = {
                    'metadata': {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'websites': urls
                    },
                    'comparison': comparison
                }
                
                # Convert to JSON
                json_data = json.dumps(export_data, indent=2)
                
                # Provide download button
                st.download_button(
                    label="Download Comparison Report",
                    data=json_data,
                    file_name=f"website_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Enhance sidebar with history and features
with st.sidebar:
    st.markdown("### 📊 Dashboard")
    
    # Create tabs in sidebar
    sidebar_tabs = st.tabs(["History", "Features", "About"])
    
    with sidebar_tabs[0]:
        st.subheader("Recent Scrapes")
        if st.session_state.history:
            # Show recent history with the newest first
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                with st.container():
                    st.markdown(f"<div class='history-item'>", unsafe_allow_html=True)
                    st.markdown(f"**{item['title'][:40]}{'...' if len(item['title']) > 40 else ''}**")
                    st.markdown(f"URL: {item['url'][:30]}...")
                    st.markdown(f"Words: {item['word_count']} • Sentiment: {item['sentiment']} • {item['timestamp']}")
                    
                    if st.button(f"Load This Analysis", key=f"load_{i}"):
                        # Load saved analysis
                        try:
                            with open(item['filename'], 'r') as f:
                                loaded_data = json.load(f)
                                st.session_state.loaded_analysis = loaded_data
                                st.session_state.loaded_url = item['url']
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading analysis: {e}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No history yet. Scrape a website to begin building your history.")
        
        # Option to clear history
        if st.session_state.history and st.button("Clear History"):
            st.session_state.history = []
            if os.path.exists('scraped_data/history.json'):
                os.remove('scraped_data/history.json')
            st.success("History cleared.")
            st.rerun()
    
    with sidebar_tabs[1]:
        st.markdown("### Key Features")
        st.markdown("""
        ✅ **Content Extraction**
        - Smart detection of main content
        - Removal of ads and navigation
        - Clean text extraction
        
        ✅ **Advanced Analysis**
        - Word frequency analysis
        - Word cloud visualization
        - Sentiment analysis
        - Readability scoring
        
        ✅ **Data Collection**
        - Extract links and images
        - Multi-page scraping
        - Website comparison
        - Named entity recognition
        - Save results for later analysis
        """)
        
        # Add a usage count
        st.session_state.scrape_count += 1
        st.metric("Scrapes Today", st.session_state.scrape_count)
    
    with sidebar_tabs[2]:
        st.markdown("### About WebSight")
        st.info("""
        **WebSight** is an advanced web scraping and content analysis tool.
        
        This tool helps you extract valuable information from websites and perform detailed analysis on the content.
        
        **Usage:**
        1. Enter a URL (including http:// or https://)
        2. Select optional settings
        3. Click "Scrape Website"
        4. Explore the extracted content through the tabs
        
        **Remember:**
        Always respect websites' terms of service and robots.txt when scraping content.
        """)
        
        st.markdown("### Limitations")
        st.warning("""
        - Some websites may block web scraping
        - JavaScript-heavy sites may have limited extraction
        - Very large websites may time out
        - Image analysis requires stable connections
        """)
        
        # Add version info
        st.markdown("---")
        st.markdown("**WebSight v2.0** | Developed with ❤️ using Streamlit")