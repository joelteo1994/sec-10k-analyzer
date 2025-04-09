#HTML Parsers 
import lxml
from lxml import etree
from bs4 import BeautifulSoup #parses and extracts information from HTML/XML (web scraping )
from bs4 import builder_registry
from urllib.parse import urlparse, parse_qs
import bs4

#Others 
import time 
import re #regular expressions: used for pattern matching in strings 
from collections import defaultdict
from collections import Counter
import os #provides functions for interacting with operating system (file paths, environment variables etc)
import requests #allows sending HTTP requests to interact with web services (e.g. APIs, downloading files)
import sys #provides access to system-related information 

#Step 1: SEC 10-K Scraper 
def get_sec_filing_url(cik): 
    """Fetches latest 10-k filing URL from SEC."""
    #url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K" #CIK is the company's Central Index Key, want to insert dynamically into the URL. 
    base_url = "https://www.sec.gov"
    index_url = f"{base_url}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K" #index_url is basically the SEC search page - same as above. CIK: Central Index Key of a company, what we want to insert dynamically into the URL 

    #Headers copied directly from my browser -> go to Developer Tools -> Networks -> command R to reload the page 
    headers = {
        "authority": "www.sec.gov",
        "method": "GET",
        "path": f"/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K",
        "scheme": "https",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"99\", \"Google Chrome\";v=\"133\", \"Chromium\";v=\"133\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    }

    # Cookies copied directly from my browser 
    cookies = {
        "_ga": "GA1.1.1724910803.1741772237",
        "nmstat": "2f5582a3-40a7-5133-dcdd-22301f13e392",
        "ak_bmsc": "B5717495A11EAAEA1337828BD1CFBEB6~000000000000000000000000000000~YAAQXZ42F+P9pE2VAQAAuyu3iRsNDBbv3B1KCig3ZlEIa14feslTb5KZu8xQRUCSwJxFlMAKoU0HWbZUFsgMoW/AU0cHd1m/JC4LXRYYNEmu6D7+bn3g3D132XPPipLq/MFAyAgonUd4o9Y3ys5wk5/m+YpiucO3qvlSZCM92u73gZjDAHS8Je6J2H1yE/jC4rA8MM93lHEWHz1eAYfRpsq0RkVk2hrOdbud76ahMO8wiv8a+y6Rw2+fdX4aRDPCAB7GiDAXHPkln8DQ8wzSBzO8KAcP/DnItq1zdCqqnuKF/+9+UdlCzpQUInAvpMR+baQ5Q6HeELeOOVXkhkUVB7Ymz5+bzjK+vd/HOinWQoa9qXrGvKycHB+NqOWgc+kzCijyA4F6zKiu5hTvpI/ZCRnWZScN1eCFldQRSP8RQrb9O6QQ/G3uaWRuzMVYSfiQ/c39glAFglw=",
        "bm_sv": "B36795B5E483E566F8AFD8699AC6950B~YAAQXZ42F43epU2VAQAAb4nBiRtRuUI83BL7aOFk70Q5MbUhrTdb6OtHT4/v3aprM16767RRv/bY/WZNa6KPiNY9X7/TdhfTIzbrfN9DFrIxW5yL3zM8XtBMtiTZNAG41F4lI35MzBnRz9R90oPXhI2BvEhTIKmwyrkl/FpBUYHTgXu1fXckWV7uUgXwNrYq5/9PU2zASuyW+AnXEFnuAPlPfhUmtHOh0aJ5cdgbxom3Z4gA6JwHcrOGaMoS~1",
        "_ga_CSLL4ZEK4L": "GS1.1.1741772237.1.1.1741772917.0.0.0",
        "_ga_300V1CHKH1": "GS1.1.1741772237.1.1.1741772919.0.0.0"
    }

    session = requests.Session() #Maintain a session to hopefully avoid SEC blocking 
    session.headers.update(headers)

    time.sleep(5) #Add delay to avoid SEC blocking

    response = session.get(index_url, cookies=cookies) #obtain response object from url via get request
    print("Status Code:", response.status_code)
    soup = BeautifulSoup(response.text) #parse the response with BeautifulSoup to facilitate data extraction 

    # Step 1: Find the link to the most recent 10-k index page, i.e. index_link 
    index_link = None 
    for link in soup.find_all("a", href=True): #find all <a> links (anchor) that contain filings; the SEC's 10-K filing links are contained in <a> tags
        if "Archives" in link["href"] and "-index.htm" in link["href"]: #SEC 10-K filings are stored in Archives; append "https://www.sec.gov" to complete the full link
            index_link = f"{base_url}{link['href']}"
            break 
    
    if not index_link: 
        print("No 10-K index page found.")
        return None 

    print(f"Found 10-K index page: {index_link}")

    #Step 2: Once inside the index page, we extract the real 10-K document 
    response = session.get(index_link)
    if response.status_code != 200: 
        print(f" Failed to fetch SEC index page. Status Code: {response.status_code}")
    
    soup = BeautifulSoup(response.text)

    #Find the inline XBRL document (`ix?doc=` link)
    for link in soup.find_all("a", href=True): 
        if "ix?doc=" in link["href"]: 
            actual_10k_url = f"{base_url}{link['href']}"
            print(f"Found actual 10-K filing: {actual_10k_url}")
            return actual_10k_url
    
    print("No Inline XBRL 10-K filing found.")
    return None

#Step 2: Extract real 10_K URL (as the above function only loads the XBRL viewer which is impossible to scrape)
def extract_real_10k_url(url): 
    """Extracts the actual 10-K filing URL from SEC's XBRL Viewer Javascript"""
    
    # if the URL is already a direct filing, return it immediately 
    if "/Archives/edgar/data/" in url and "ix?" not in url: 
        print(f"Using direct 10-K filing: {url}")
        return url 
    
    # if URL is in an XBRL viewer ('ix?doc=...'), extract the real 10-K URL 
    parsed_url = urlparse(url)
    if parsed_url.path =="/ix": 
        query_params = parse_qs(parsed_url.query)
        if "doc" in query_params: 
            real_10k_url = f"https://www.sec.gov{query_params['doc'][0]}"
            print(f"Extracted real 10-K URL from XBRL Viewer: {real_10k_url}")
            return real_10k_url

    print("No valid 10-K link found.")
    return None

#Step 3: Download the actual SEC filing 
def download_sec_filing(url, company_name): 
    """Downloads SEC 10-K filing (HTML, PDF, or Text) dynamically."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Referer": "https://www.sec.gov/",
        "DNT": "1",
        "Connection": "keep-alive",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Host": "www.sec.gov"
    }

    session = requests.Session() 
    session.headers.update(headers)

    time.sleep(5)

    actual_10k_url = extract_real_10k_url(url)
    if not actual_10k_url: 
        print("No valid 10-K URL extracted")
        return None 
    

    response = session.get(actual_10k_url, allow_redirects=True)
    print(f"Response Headers: {response.headers}")

    if response.status_code != 200: 
        print(f"Failed to download SEC filing: Status Code {response.status_code}")
        return None 

    #Extract year dynamically from URL 
    match = re.search(r"(\d{4})", url) #look for a 4-digit year
    year = match.group(1) if match else "unknown"

    #Create a downloads folder if does not exist 
    os.makedirs("downloads", exist_ok=True)

    file_path = os.path.join("downloads", f"{company_name}_10k_{year}.html")

    # Save the file
    with open(file_path, "wb") as file: #use wb for PDFs and text files 
        file.write(response.content)
    
    print(f"10-k filing saved as: {file_path}")
    return file_path 

#Step 5: test to check that scraper works as expected, okay to hardcode specific cik 
if __name__ == "__main__": #i.e. tells python to run this test if I run this file directly from command line. 
    cik = "0000789019" #Microsoft Inc 
    company = "Microsoft"
    url = get_sec_filing_url(cik)
    if url: 
        print(f"Successfully retrieved 10-K filing URL: {url}")
        path = download_sec_filing(url, company) #extract_real_url function embedded in download_sec_filing function
        print("Downloaded to:", path)
    else: 
        print("No 10-K filing found")
    
    #to test: just type " python utils/scrape_sec.py" in command line, it will trigger the if __name__ == "__main_-" code
    
    