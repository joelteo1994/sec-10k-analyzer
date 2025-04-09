import sys #provides access to system-related information 

#HTML Parsers 
import lxml
from lxml import etree
from bs4 import BeautifulSoup #parses and extracts information from HTML/XML (web scraping )
from bs4 import BeautifulSoup
from bs4 import builder_registry
from urllib.parse import urlparse, parse_qs
import bs4

#Data Structures 
import pandas as pd #provides data structures (DataFrame, Series) for efficient data 
import numpy as np #support for large multi-dim arrays

#Others 
import time 
import re #regular expressions: used for pattern matching in strings 
from collections import defaultdict
from collections import Counter
import os #provides functions for interacting with operating system (file paths, environment variables etc)

# Step 1: Extract Plain Text from the Filing 
def extract_text_from_10k(file_path): 
    """Extracts readable text from a downloaded SEC 10-K HTML filing"""
    with open(file_path, "r", encoding="utf-8") as file: 
        soup=BeautifulSoup(file, "html.parser") #creates HTML tree in soup object which we can traverse 
    
    #Remove script, style, and unwanted elements 
    for tag in soup(["script", "style", "meta", "noscript", "link", "iframe"]): 
        tag.decompose()

    #Remove inline formatting (e.g.. <span>)
    for span in soup.find_all("span"): 
        span.unwrap() #removes span formatting elements (stylistics) but keeps only the text 


    #Extract only meaningful text in block structure (using the <div> and <p> tags)
    text_blocks = [tag.get_text(separator=" ", strip=True) for tag in soup.find_all(["div", "p"])] #extract all visible text from <div> and <p> tags
    full_text = "\n".join(text_blocks) #join everything into a single text document 

    #Clean text to remove multiple spaces, non-ASCII characters, and extra lines 
    full_text = re.sub(r'\s+', ' ', full_text)  # Replace multiple spaces with single space
    full_text = re.sub(r'[^\x00-\x7F]+', '', full_text)  # Remove non-ASCII characters
    full_text = full_text.strip()  # Trim leading/trailing whitespace

    return full_text 

#Step 2: Detect Section Headers 
def detect_section_headers(text): 
    """
    Dynamically detects section headers from the 10-K Table of Contents by 
    - Detecting "Item X. <Title>" patterns
    - Ensuring each section is individually stored
    """
    # 🔍 Step 1: Find all "Item X. <Title>" matches
    section_matches = re.findall(r"(Item\s+\d+[A-Z]?\.\s+[\w\s&-]+)", text, re.IGNORECASE) #matches "item" (case insensitive), then number with optional alphabet, then period (.), then ensure space after period, and actual section title (words, spaces, ampersands, and dashes)

    # 🔴 Debugging: Print found matches
    #print("\n🔍 Step 1: Raw Section Matches Found")
    #for match in section_matches:
    #    print(f"➡️ {match}")

    # If nothing is found, stop here
    if not section_matches:
        print("⚠️ No sections detected! The regex might be incorrect or the text format is unexpected.")
        return {}

    
    # Step 2: Process each detected section separately
    cleaned_sections = {}

    for match in section_matches:
        # 🔍 Extract the "Item X." portion
        item_number = re.match(r"(Item\s+\d+[A-Z]?)\.", match)
        
        if item_number:
            item_number = item_number.group(1)  # Extract just "Item X"
            section_title = match.replace(item_number + ".", "").strip()  # Remove "Item X." from title

            # Remove everything from "The" onward
            section_title = re.split(r"\bThe\b", section_title, 1, flags=re.IGNORECASE)[0].strip()
            
            # 🔹 Store each section individually
            cleaned_sections[item_number] = section_title
    
    sorted_cleaned_sections = dict(sorted(cleaned_sections.items(), key=lambda x:(int(re.search(r'\d+', x[0]).group()), x[0])))

    # 🔍 Step 4: Print final sorted sections
    #print("\n✅ Step 4: Final Sorted Cleaned Sections")
    #for section, title in sorted_cleaned_sections.items():
    #    print(f"📌 {section}: {title}")

    return sorted_cleaned_sections

def normalize(s): 
    return re.sub(r'[^a-z0-9]', '', s.lower())

#Alternative to Step 1 and 2: 
#Key insight: working in HTML space allows for presence of more unique identifiers; if we strip away the tags we are left with just a word blob which is difficult to distinguish diff instances of Item X: Title appearing
def detect_section_headers_with_content(file_path):
    """
    Extracts robust section headers and their associated content from a 10-K HTML file.

    - Uses span and <p> tags to detect bolded section headers.
    - Matches only main items (e.g. 'Item 1', not 'Item 1A').
    - Captures full HTML content between each section.
    
    Returns:
        dict: { "Item X": { "title": ..., "content": ... } }
    """
    from bs4 import BeautifulSoup
    import re

    with open(file_path, "r", encoding='utf-8') as file: 
        soup = BeautifulSoup(file, 'html.parser')

    headers = {}

    # === Pass 1: Find <span> tags with bold font-weight
    span_tags = soup.find_all("span")
    for span in span_tags:
        style = span.get("style", "")
        text = span.get_text(strip=True)

        if "font-weight:bold" in style or "font-weight:700" in style:
            match = re.match(r'(Item\s*\d+[A-Z]?)\.?\s*[:\-–.]?\s*(.+)', text, re.IGNORECASE)
            if match:
                item = match.group(1).title().strip()
                title = match.group(2).strip()
                if re.fullmatch(r'Item\s+\d+', item, re.IGNORECASE):
                    headers[item] = {"title": title, "tag": span}

    # === Pass 2: Look inside <p> tags for combined bold spans (Microsoft style) where it is <span ITEM 1. B</span><span>USINESS</span> 
    p_tags = soup.find_all("p") #find all <p> paragraph tags in the HTML using BS, store them in p_tags
    for p in p_tags: #loop over each <p> tag one by one
        #Combine all bold spans into one string
        bold_spans = p.find_all("span", style=re.compile("font-weight:bold|font-weight:700", re.IGNORECASE)) #inside current <p>, find all <span> tags with CSS style that includes bold formatting 
        if bold_spans and len(bold_spans) >= 1: #proceed only if there's at least one bold <span> found 
            full_text = ""
            for i, span in enumerate(bold_spans):
                text = span.get_text(strip=True)
                if i > 0:
                    prev = bold_spans[i - 1].get_text(strip=True)
                    if not (prev[-1:].isalpha() and text[:1].isalpha()):
                        full_text += " "
                full_text += text
            full_text = full_text.strip()
            #print(bold_spans)
            #full_text = " ".join(span.get_text(strip=True) for span in bold_spans).strip() #concatenate text from all the bold spans into one string, with spaces between them. Also strips any leading/training whitespace
            full_text = re.sub(r"\s+", " ", full_text)
            while re.search(r'\b([A-Z])\s+([A-Z])', full_text):
                full_text = re.sub(r'\b([A-Z])\s+([A-Z])', r'\1\2', full_text)
            
            #print(full_text)

            match = re.match(r'(Item\s*\d+[A-Z]?)\.?\s*[:\-–.]?\s*(.+)', full_text, re.IGNORECASE)
            if match:
                #print(f"\n✅ Matched full_text: {full_text}")
                #print(f"📌 Group 1 (item): {match.group(1)}")
                #print(f"📌 Group 2 (title): {match.group(2)}")
                item = match.group(1).title().strip()
                title = match.group(2).strip()
                while re.search(r'\b([A-Z])\s+([A-Z])', title):
                    title = re.sub(r'\b([A-Z])\s+([A-Z])', r'\1\2', title)
                if re.fullmatch(r'Item\s+\d+', item, re.IGNORECASE):
                    headers[item] = {"title": title, "tag": p}
    
    # === Pass 3: Also scan <div> tags for bold spans (Meta-style)
    div_tags = soup.find_all("div")
    for div in div_tags:
        bold_spans = div.find_all("span", style=re.compile("font-weight:bold|font-weight:700", re.IGNORECASE))
        if bold_spans and len(bold_spans) >= 1:
            full_text = ""
            for i, span in enumerate(bold_spans):
                text = span.get_text(strip=True)
                if i > 0:
                    prev = bold_spans[i - 1].get_text(strip=True)
                    if not (prev[-1:].isalpha() and text[:1].isalpha()):
                        full_text += " "
                full_text += text
            full_text = full_text.strip()
            full_text = re.sub(r"\s+", " ", full_text)
            while re.search(r'\b([A-Z])\s+([A-Z])', full_text):
                full_text = re.sub(r'\b([A-Z])\s+([A-Z])', r'\1\2', full_text)

            match = re.match(r'(Item\s*\d+[A-Z]?)\.?\s*[:\-–.]?\s*(.+)', full_text, re.IGNORECASE)
            if match:
                item = match.group(1).title().strip()
                title = match.group(2).strip()
                while re.search(r'\b([A-Z])\s+([A-Z])', title):
                    title = re.sub(r'\b([A-Z])\s+([A-Z])', r'\1\2', title)
                if re.fullmatch(r'Item\s+\d+', item, re.IGNORECASE):
                    headers[item] = {"title": title, "tag": div}

    # === Extract content between tags
    items_sorted = sorted(headers.items(), key=lambda x: int(re.search(r'\d+', x[0]).group()))
    results = {}

    for i in range(len(items_sorted)):
        item, meta = items_sorted[i]
        start_tag = meta["tag"]
        end_tag = items_sorted[i + 1][1]["tag"] if i + 1 < len(items_sorted) else None

        collected = []
        for sibling in start_tag.find_all_next():
            if end_tag and sibling == end_tag:
                break
            collected.append(sibling)

        # Join HTML, parse once, then clean
        joined_html = "".join(str(tag) for tag in collected)
        soup_section = BeautifulSoup(joined_html, "html.parser")

        # Final clean text, line-separated
        plain_text = soup_section.get_text(separator="\n", strip=True)
        lines = plain_text.split("\n")
        # Combine and lowercase for fuzzy comparison
        header_line = normalize(meta["title"])
        first_lines = normalize(" ".join(lines[:2]))
        # If the first two lines repeat the header, drop them
        if header_line and header_line in first_lines:
            print(f"🧹 Removing repeated header in '{item}'")
            lines = lines[2:] if len(lines) > 2 else lines[1:]
        if lines and re.match(r"^Item\s+\d+[A-Z]?\.", lines[-1], re.IGNORECASE):
            lines = lines[:-1]
        plain_text = "\n".join(lines).strip()


        # Optional: remove exact duplicates across lines
        seen = set()
        deduped_lines = []
        for line in plain_text.splitlines():
            if line not in seen:
                deduped_lines.append(line)
                seen.add(line)

        results[item] = {
            "title": meta["title"],
            "content": "\n".join(deduped_lines)
        }

    return results

# Step 3: Remove TOC for cleaner REGEX search 
def remove_toc(text, section_headers):
    """
    Removes everything before the first detected section header.
    
    - Dynamically detects the first real content section.
    - Ensures TOC references are removed without hardcoding.
    """

    if not section_headers:
        print("⚠️ No section headers detected! Returning full text.")
        return text  # If no headers detected, return original text

    # 🔍 **Step 1: Find the first actual section header**
    first_section_key = next(iter(section_headers))  # Get first key from the sorted dictionary
    first_section_value = section_headers[first_section_key]  # Get corresponding value
    first_section_full = f"{first_section_key}. {first_section_value}"  # Concatenate
    pattern = re.escape(first_section_full)  # Escape special regex characters
    #print(f"First Section: {first_section_full}")

    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        print(f"⚠️ '{first_section_full}' not found in text! Returning full document.")
        return text  # If first section isn't found, return original text

    clean_start = match.start()  # Index where real content starts
    print(f"\n🔎 TOC detected. Removing all text before index: {clean_start} ({first_section_full})")

    # ✂️ **Step 2: Remove everything before this index**
    clean_text = text[clean_start:].strip()

    # 🧐 **Step 3: Debugging - Show what's being removed**
    print("\n🗑 **TOC Snippet Being Removed (First 1000 chars):**\n")
    print(text[:clean_start][:1000])  # Show first 1000 chars of removed TOC for review

    print("\n🚀 **TOC removal complete!** ✅\n")
    print(f"✅ Cleaned document starts at character index {clean_start}.")
    print(f"📌 Cleaned document length: {len(clean_text)} characters")
    print(f"Cleaned Text: {clean_text[:500]}")

    return clean_text

#Step 4: Extract main sections for text analytics 
def extract_main_sections(text, max_items=16):
    """
    Extracts the first N major sections from a 10-K filing.
    - Ignores sub-sections like "1A, 1B".
    - Ensures proper section segmentation.
    """

    # 🔍 Step 1: Detect section headers dynamically
    section_headers = detect_section_headers(text)  # Get cleaned, sorted headers
    print(section_headers)

    if not section_headers:
        print("No valid section headers found!")
        return {}

    # 🔍 Step 2: Filter only "Item 1", "Item 2", ... (ignore "1A", "1B", etc.)
    main_sections = {k: v for k, v in section_headers.items() if re.match(r"Item\s+\d+\b", k)} #use the Regex to match only "Item 1", "Item 2" etc

    # Limit to the first `max_items`
    main_sections = dict(sorted(main_sections.items(), key=lambda x: int(re.search(r'\d+', x[0]).group()))[:max_items])

    # 🛠 Debugging: Print extracted main sections before sorting
    #print("\n🔍 Extracted Main Sections (Before Sorting):")
    #for key, value in main_sections.items():
    #    print(f"📌 {key}: {value}")

    sections = {}  # Store extracted sections
    section_positions = {}  # Store start positions of each section

    # 🔍 Step 3: Find section positions using regex (match full title exactly)
    for section, title in main_sections.items():
        pattern = rf"{re.escape(section)}\s*\.*\s*" + r"\s*".join(re.escape(word) for word in title.split()) + r"\b"
        #print(f"Searching for: {pattern}")
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
 
        if matches:
            section_positions[section] = matches[0].start()  # Pick first valid match

    # Ensure section positions are sorted by start index
    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])

    # 🔍 Step 4: Extract section text **correctly**
    for i in range(len(sorted_sections)):
        current_section, start_idx = sorted_sections[i]

        if i < len(sorted_sections) - 1:
            next_section, next_idx = sorted_sections[i + 1]
            section_text = text[start_idx:next_idx].strip()  # Extract from current title to next title
        else:
            section_text = text[start_idx:].strip()  # Last section, extract until end

        # 🚀 Clean up the extracted text
        section_text = re.sub(r"^.*?" + re.escape(current_section) + r"\s*\.*\s*" + re.escape(main_sections[current_section]), '', section_text, flags=re.IGNORECASE).strip()
        section_text = re.sub(r"\s{2,}", " ", section_text).strip()  # Remove excessive whitespace

        sections[current_section] = section_text

        # 🛠 Debugging Extraction Issues
        #if len(section_text.strip()) < 100:
        #    print(f"❌ WARNING: Section {current_section} seems too short! ({len(section_text.strip())} chars)")
        #else:
        #    print(f"Extracted Section: {current_section} (Length: {len(section_text.strip())} characters)")

    # Step 5: Print Final Sorted Order
    print("\n✅ Final Sorted Extracted Sections:")
    for sec, txt in sections.items():
        print(f"📌 {sec} (Length: {len(txt)} characters)")

    return sections

#Bous: Function that automatically detects latest 10-K HTML file 
def get_latest_10k_file(downloads_folder="downloads"): 
    """
    Retrieves most recently modified .html file from the downloads folder 
    Args: 
    - downloads_folder (str): name of downloads folder where the 10-K files are saved (default = 'downloads')
    Returns: 
    - file_path (str): Full file path to the latest 10-K HTML file, or None if no .html file is gound
    """

    import glob 
    import os

    #Grab all HTML files in the specified folder
    html_files = glob.glob(os.path.join(downloads_folder, "*.html"))
    if not html_files: 
        return None 
    latest_file = max(html_files, key=os.path.getmtime) #return latest modified file using getmtime (modification time)
    return latest_file #returns full path to the latest file

if __name__ == "__main__": 
    file_path = get_latest_10k_file()
    if not file_path: 
        print("Error: No .html files found in downloads/ folder")
        sys.exit(1) # Exit with error code 1 (i.e. something wrong)
    
    print(f"Using latest file: {file_path}")
    raw_text = extract_text_from_10k(file_path)
    headers = detect_section_headers(raw_text)
    cleaned_text = remove_toc(raw_text, headers)
    extracted_sections = extract_main_sections(cleaned_text)

    #Dynamically name output file based on input 
    base_name = os.path.basename(file_path).replace(".html", "_Extracted_Sections.txt") #generates friendly output filename e.g. "Apple_10k_3201_Extracted_Sections.txt"
    output_path = os.path.join("downloads", base_name) #keeps outputs nicely organised in the same folder

    with open(output_path, "w", encoding="utf-8") as file:
        for section, text in extracted_sections.items():
            file.write(f"📌 {section}\n")  # Section header
            file.write("=" * len(section) + "\n")  # Underline
            file.write(text + "\n\n")  # Section content + spacing

    print(f"All sections saved in: {output_path}")







