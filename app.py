#Flask Backend
from flask import Flask, render_template, request, jsonify, send_file 

#Import utility modules - separation of concerns 
from utils.scrape_sec import get_sec_filing_url, download_sec_filing
from utils.section_extractor import (
    get_latest_10k_file, 
    extract_text_from_10k, 
    detect_section_headers, 
    detect_section_headers_with_content,
    remove_toc, 
    extract_main_sections
)
from utils.nlp_analysis import (
    retrieve_corpus, 
    process_text, 
    process_text_2,
    summarize_text, 
    analyze_sentiment, 
    generate_wordcloud, 
    extract_topics_from_texts
)

from utils.quant_analysis import (
    extract_meaningful_tables, 
    elevate_headers, 
    elevate_index, 
    get_cleaned_columns, 
    finalize_table, 
    remove_row_duplicates_correctly, 
    compact_table_no_padding, 
    repair_broken_header,
    promote_index_if_row_blank, 
    promote_index_if_row_values_blank,
    plot_data,
    is_blank_header, 
    elevate_row_if_index_or_row_is_date, 
    infer_title_from_index, 
    plot_data_2, 
    finalize_headers_with_padding
)

import os #OS-level file access
import uuid #For generating unique filenames
import regex as re 

#Initialise Flask app 
app = Flask(__name__) #"__name__" tells Flask where the app is being run from, so as to know where to look for resources. I.e. use the current file's location as reference point to find templates/ and static/ folders nearby

#Default landing route
@app.route("/") #flask decorator: note: the '/' is the root url (e.g. http://localhost:5000/). 
def home(): #i.e. when someone visits webpage, call this function (i.e. render the index.html template)
    return render_template("index.html") #Loads frontend homepage which contains form for user to input CIK and company name for further analysis

#Route that handles form data submission from the front-end. Decorator tells flask to "decorate" the function below so that it becomes the handler for a specific url. This then defines a new entry pooint into my web app. 
#I.e. when request made to backend at /metadata (e.g. http://localhost:5000/metadata) and it is a POST request, Flask will run the function directly below it. 
@app.route("/metadata", methods=["POST"]) #"POST": client sending data (CIK and Coy Name) to the backend (i.e. someone submitting something to you)
def metadata():
    cik = request.form.get("cik") #retrieves data from the server: here, grabs what user typed into input fields for CIK and company (note that this is not a get request, it is just getting data from a POSTed form)
    company = request.form.get("company")
    
    if not cik or not company: #if either CIK or company name is missing, then return error 
        return jsonify({"error": "Missing CIK or company name"}), 400 #converts python dictionary into JSON response -> language that frontend expects. 400 is the HTTP status code; accessible via response.status == 400
        #dictionaries easier to jsonify, because both are inherently key-value structures 

    #Pass the cik into our utility functions to get url, download sec filing and extract sections 
    url = get_sec_filing_url(cik) 
    file_path = download_sec_filing(url, company)
    extracted_sections = detect_section_headers_with_content(file_path) #returns dict within dict: outer dict is "Item 1:" then innner dict is {"title": "Business", "content": "..."}

    # Extract proper titles from headers through dictionary comprehension. Format: ("what we want to store as combination of units" for units in iterables if condition is true)
    section_titles = {
        k: v["title"] #recall: k is "Item X", v is a dict {"title": "Business", "content: "..."}
        for k, v in extracted_sections.items()
        if v["title"].strip() and not v["title"].lower().startswith("none") #i.e. if title is not empty (if v["title"].strip() returns False otherwise) and dosent' start with none
        #in Python, the following values are "falsy" i.e. evaluate to False in a conditional like if: "" (empty string), 0, None, []/{}/() (i.e. empty iterables)
    }

    #Deduplication logic 
    seen_titles = set() #stores processed raw strings stripped of any special characters, punctuations, spaces; anything outside [a-z] or 0-9
    deduped_section_titles = {}

    for k, title in section_titles.items(): #iterate thru each section_title key value pair
        norm_title = re.sub(r'[^a-z0-9]', '', title.lower()) #title.lower() converts title to lower case, and the earlier regex pattern matches any character not (^) in the range of [a-z] or 0-9. re.sub removes all such characters
        if norm_title not in seen_titles:
            deduped_section_titles[k] = title #capitalise each word
            seen_titles.add(norm_title) #store normalised version in seen_titles (using seen_titles.add(norm_title))

   #***Debugging portion***
    for k in section_titles:
        match = re.search(r'\d+', k)
        print(f"🔍 Key: '{k}' → Number extracted: {match.group() if match else '❌ NO MATCH'}")
    #******

    #*****Quant Section ********
    #Need to go thru the whole table cleaning steps since the process of retrieving the title is embedded within the table cleaning 
    raw_tables = extract_meaningful_tables(file_path) #retrieve tables
    filtered_tables = {} #first dictionary to store the actual filtered tables, but not explicitly returned in this metadata route (since we just need the table titles (cleaned)). I.e. separating display logic vs data payload 
    for table_name, table_info in raw_tables.items(): #recall table_info contains ['title'] and ['data'], latter is the actual df
        df = table_info["data"]
        print(f"\n Processing {table_name}: {table_info['title']}")

        if df is not None and not df.empty:
            df = elevate_headers(df)
            df = elevate_index(df)
            cleaned_columns = get_cleaned_columns(df)
            df = finalize_table(df, cleaned_columns)
            df = remove_row_duplicates_correctly(df)

            filtered_tables[table_name] = {
                "title": table_info['title'],
                "data": df
            }

    for table_name, table_info in filtered_tables.items():
        df = table_info['data']
        if df is not None and not df.empty:
            print(f"\n Compacting Table {table_name}: {table_info['title']}")
            df_compacted = compact_table_no_padding(df)
            filtered_tables[table_name]['data'] = df_compacted
            print(df_compacted.head())
        else:
            print(f"Skipping {table_name}: Empty DataFrame")
    
    #Store table_titles in a dictionary
    table_titles = {}
    for table_name, table_info in filtered_tables.items():
        df = table_info['data']
        title = table_info['title']

        if df is not None and not df.empty:
            print(f"Finalising table format {table_name}: {title}")

            df = repair_broken_header(df)
            df = elevate_row_if_index_or_row_is_date(df)
            df, updated_title = promote_index_if_row_blank(df, title)
            df, updated_title = promote_index_if_row_values_blank(df, updated_title)
            df = elevate_row_if_index_or_row_is_date(df)
            final_title = infer_title_from_index(df, updated_title)

            filtered_tables[table_name]['data'] = df
            filtered_tables[table_name]['title'] = final_title

            table_titles[table_name] = final_title.title()
            print(df.head())
        else:
            print(f"⚠️ Skipping {table_name}: Empty DataFrame")

    return jsonify({ #single JSON object with two keys 
        "sections": deduped_section_titles, #first key stores the deduplicated form 10-K section titles (e.g. "Item 1": "Business")
        "tables": table_titles #second key stores the table titles (e.g. "Table_1": "Consolidated Balance Sheet")
    })


#Route for processing user-submitted CIK and company (main-logic engine)
@app.route("/analyze", methods=['POST']) #flask route decorator: "/analyse" is the custom endpoint where app listens for analysis request. I.e. when some visits or submits to "/analyze", this function is called
def analyze(): #note: by default, Flask routes only accept GET request (e.g. visiting by clickling link or reloading page). But here, accepting form submission in the front-end, so backend needs to be ready to say I am ready to receive data, not just serve HTML
    cik = request.form.get("cik") #Extract CIK that user filled into the form 
    company_name = request.form.get("company") #Extract company name that user filled into the form
    selected_sections = request.form.getlist("sections") #get user-selected sections for NLP
    selected_tables = request.form.getlist("tables") #get user-selected tables for quant 

    if not cik or not company_name: 
        return jsonify({"error": "Missing CIK or company name."}), 400 # Input validation 
    
    #Step 1: Scrape SEC
    url = get_sec_filing_url(cik) #Step 1: Scrape SEC for latest 10-K link 
    if not url: 
        return jsonify({"error": "10-K filing not found"}), 404

    #Step 2: Download form 10-k in html 
    file_path = download_sec_filing(url, company_name) #Step 2: Download the 10-K file
    if not file_path: 
        return jsonify({"error": "Failed to download 10-K"}), 500
    
    #Step 3: Extract and clean sections 
    extracted_sections = detect_section_headers_with_content(file_path) #returns dictionary of dictionary 

    #Step 4: NLP Analysis 
    available_section_keys = list(extracted_sections.keys()) #extracts all the items and puts in a list ["Item 1", "Item 2", ..., "Item 16"]
    if not selected_sections: 
        selected_sections = [s for s in available_section_keys if s in ["Item 1", "Item 7"]]

    #******* Debugging Section ******************************************
    print(f"🧠 Available sections: {available_section_keys}")
    print(f"📌 Selected sections for NLP: {selected_sections}")

    for sec in selected_sections:
        if sec in extracted_sections:
            print(f"\n📄 {sec} - {extracted_sections[sec]['title']}")
            preview = extracted_sections[sec]['content'][:300]
            print(f"🧾 Preview content: {preview}...\n{'-'*50}")
        else:
            print(f"⚠️ WARNING: Section '{sec}' not found in extracted_sections")
    #**********************************************************************

    text_for_analysis = retrieve_corpus(extracted_sections, selected_sections)
    
    #******* Debugging Section ******************************************
    print(f"📦 retrieve_corpus() returned type: {type(text_for_analysis)}")
    if isinstance(text_for_analysis, str):
        print(f"🧾 Length of joined string: {len(text_for_analysis)} chars")
        print(f"🔍 Sample: {text_for_analysis[:300]}")
    elif isinstance(text_for_analysis, list):
        print(f"📑 Number of docs: {len(text_for_analysis)}")
        print(f"🧾 First doc sample: {text_for_analysis[0][:300]}")
    #**********************************************************************

    word_freq, _ = process_text(text_for_analysis) #Word frequency analysis 

    #Step 5: Generate word cloud 
    wordcloud_path = f"static/wordcloud_{uuid.uuid4().hex}.png" #unique filename 
    generate_wordcloud(word_freq, output_path=wordcloud_path)

    #Step 6: Summarisation and Sentiment 
    summary = summarize_text(text_for_analysis, summarization_type="extractive", max_length=250, min_length=50, overlap=100) #Auto-generated summary tables, but in df format. Need to change to html
    sentiment_summary_df, sentiment_detail_df = analyze_sentiment(text_for_analysis, method="nltk", batch_size=5) #analyze_sentiment returns both a summarised version and a detailed version of the sentiment analysis
    sentiment_summary_html = sentiment_summary_df.to_html(classes="table table-bordered table-sm", index=False) # which we then convert into html format to be rendered in the front end
    sentiment_detail_html = sentiment_detail_df.to_html(classes="table table-striped table-sm", index=False)

    #Step 7: Topic Modelling 
    topic_df = extract_topics_from_texts(text_for_analysis) #BERTopic extraction 
    topics_csv = f"static/topics_{uuid.uuid4().hex}.csv" #save to csv for download
    topic_df.to_csv(topics_csv, index=False)

    #Step 8: Quantitative Table Extraction 
    raw_tables = extract_meaningful_tables(file_path)
    processed_tables = {}
    chart_paths = []  # To store chart image paths

    for table_name, table_info in raw_tables.items(): 
        df = table_info["data"]
        title = table_info['title']
        if df is not None and not df.empty: 
            #Elevate index and headers
            df = elevate_headers(df)
            df = elevate_index(df)
            #Get clean columns and finalise the table
            cleaned_cols = get_cleaned_columns(df)
            df = finalize_table(df, cleaned_cols)
            #Remove duplicated rows and compact table
            df = remove_row_duplicates_correctly(df)
            df = compact_table_no_padding(df)
            #Readjust headers and titles 
            df = repair_broken_header(df)
            df = elevate_row_if_index_or_row_is_date(df)
            df, updated_title = promote_index_if_row_blank(df, title)
            df, updated_title = promote_index_if_row_values_blank(df, updated_title)
            df = elevate_row_if_index_or_row_is_date(df)
            df = finalize_headers_with_padding(df)
            final_title = infer_title_from_index(df, updated_title)

            # Save table
            processed_tables[table_name] = {
                "title": final_title, 
                "data": df, 
                'html': df.to_html(classes="table table-striped table-hover table-sm table-bordered align-middle",
                                    border=0,
                                    justify="center",
                                    index=True,
                                    escape=False)

            }

    if not selected_tables: 
        selected_tables = [t for t in processed_tables.keys() if t in ["Table_7", "Table_8"]]

    #Call generalised plot function after processing tables 
    chart_paths = plot_data(processed_tables, selected_tables, save_dir="static")

    # Overwrite the DataFrame with its HTML so front-end gets it nicely
    for k in processed_tables:
        processed_tables[k]["data"] = processed_tables[k]["html"]
        del processed_tables[k]["html"]  # clean up

    # Preview top topics (for display)
    topics_preview = topic_df["Name"].head(10).tolist() if not topic_df.empty else []

    #Step 9: return JSON response (flask backend prepares ingredients and dishes in form of JSON, then frontend HTML and JavaScript brings dish to user as the waiter)
    return jsonify({
        "message": "Analysis complete.",
        "sections_used": selected_sections,
        "wordcloud": wordcloud_path,
        "summary": summary,
        "sentiment_summary": sentiment_summary_html,
        "sentiment_details": sentiment_detail_html,
        "topics_csv": topics_csv,
        "topics_preview": topics_preview,
        "tables": { k: v for k, v in processed_tables.items() if k in selected_tables },
        "graphs": chart_paths
    })

# === Optional route to download the CSV ===
@app.route("/download/topics") #sets up a new route; when browser visits this URL with a query param, Flask will call the function to download topics
def download_topics():
    path = request.args.get("path")  # Extract path from query
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)  # Return file to user

# === Run server in debug mode if executed directly ===
if __name__ == "__main__": #call in terminal using python app.py (under which Python sets special variable __name__ to "__main__"), but won't run if imported directly 
    app.run(debug=True)  # Start Flask app with hot-reload