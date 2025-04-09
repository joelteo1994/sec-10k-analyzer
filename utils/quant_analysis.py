#Data Structures 
import pandas as pd #provides data structures (DataFrame, Series) for efficient data 
import numpy as np #support for large multi-dim arrays

#Others 
import sys
import time 
import re #regular expressions: used for pattern matching in strings 
from collections import defaultdict
from collections import Counter
import os #provides functions for interacting with operating system (file paths, environment variables etc)

from bs4 import BeautifulSoup

def is_numeric(value):
    """Check if a value contains numeric data."""
    try:
        float(value.replace(",", "").replace("$", ""))  # cleaning step: Remove commas and dollar signs before conversion and then tries to convert it to a float
        return True #if successful, return True
    except ValueError: #e.g. float('abcde') will give ValueError. I.e. can only convert pure numeric strings and scientific notation (e.g. "1e3")
        return False

def clean_value(value): 
    "Remove all dollar signs and percentage signs"
    value = value.replace(",", "").strip()  # Remove commas
    value = re.sub(r"^\$", "", value)  # Remove leading $ only. re.sub(pattern, replacement, value) tells python to find things that match pattern in value, and replace them with replacement. Here, it is find leading dollar signs and replace them with nothing
    value = value.replace("%", "")  # Remove % but keep numeric format
    return value.strip()

def extract_meaningful_tables(file_path, min_numeric_cells=2, max_lookups = 5):
    """Extracts only meaningful tables from an SEC filing HTML file, and removes duplicate entries in rows (except first column)"""

    #Opens the file using the filepath provided and parses into the "soup" object using the BeautifulSoup parser 
    with open(file_path, "r", encoding="utf-8") as file: #"r" is read mode. with open(...) ensures file is automatically closed afterward, encoding="utf-8" handles most modern text properly
        soup = BeautifulSoup(file, "html.parser") #parses html into a navigable tree (i.e. DOM or Document Object Model). Tree-like memory rep of HTML doc; how a broser or parser sees and navigates HTML structure 

    extracted_tables = {}  # Store extracted tables
    table_count = 1  # Ensure consistent numbering

    # Locate all tables in the HTML document
    tables = soup.find_all("table") #returns a list of all <table> tags, where each element in the list is a BS Tag object representing an entire <table>...</table> block, including 
    #all the rows <tr>, all the cells <td> or <th> and any nested elements inside the cells 

    for table_idx, table in enumerate(tables):
        #for each table, create empty lists to store headers, table_data etc
        table_data = []
        headers = []
        table_title = None
        table_subtitle = None 

        # First, check whether the table contains numbers using the below numeric filter logic 
        numeric_count = sum(
            is_numeric(cell.get_text(strip=True)) #generator expression that iterates thru all rows and cells in the current table, and for each cell, extract the text without spaces and checks if it is numeric, returns True (1) or False (0)
            for row in table.find_all('tr') 
            for cell in row.find_all('td')
        )

        if numeric_count < min_numeric_cells: 
            print(f"⏭️ Skipping Table {table_idx+1}: Only {numeric_count} numeric cells.")
            continue

        # Improved logic to extract table title and subtitle
        #First, start at the current table and assume we haven't found a title or subtitle 
        current = table
        title_found = False
        subtitle_found = False
        steps = 0

        #First pass: try to look at previous siblings up to limit defined by max_lookups. 
        while current and steps < max_lookups: #moves up the HTML tree sibling by sibling, up till max_lookup steps back to avoid going too far
            current = current.find_previous_sibling() #current is set to be a BS Tag object representing an actual HTML element (e.g. <p>, <div> <h2> etc)
            steps += 1
            if current and current.name == "table": #stops if another table is hit (to avoid crossing unrelated sections)
                break
            #Logic to check if current element looks like a title/subtitle
            if current and current.name in ["div", "p", "span", "h2", "h3", "h4"]: #restrict to tags that could logically contain titles. current.name gives tag name as a string e.g. current.name == "h2" checks if the tag is a <h2> tag
                text = current.get_text(strip=True)
                if not text:
                    continue

                #Extract styles as further indicators of title e.g. in <p style="font-weight:700; color:red;" align="center">Balance Sheet</p>
                style = current.get("style", "") if current.has_attr("style") else "" #retrieves the style attribute of the tag if available, else nothing. This retrieves "font-weight:700: color:red"
                align = current.get("align", "") if current.has_attr("align") else "" #retrieves the align -> "center"

                #Logic block for title -> refine this when see more form 10ks with diff stylistic elements 
                if not title_found and (
                    current.name in ["h2", "h3", "h4"]
                    or "underline" in style
                    or "text-align:center" in style
                    or "center" in align
                    or "font-weight:700" in style
                ): #need semicolon as this is the end of a complex "if" statement 
                    #if len(text) < 100: 
                    table_title = text
                    title_found = True
                elif not subtitle_found and (
                    "font-weight:600" in style
                    or "font-weight:700" in style
                    or current.name in ["p", "span"]
                ):
                    table_subtitle = text
                    subtitle_found = True

            if title_found and subtitle_found:
                break

        #Second pass if not able to find good title within the same parent: look at other parents
        if not title_found:
            outer = table.parent #first, go up to the parent of the table
            steps = 0
            max_lookups = 6  # reuse or redefine as needed

            while outer and steps < max_lookups:
                outer = outer.find_previous() #then, walk upward/backward in the DOM (not just siblings)
                steps += 1

                if not outer or outer.name == "table": #Bail if hit another table
                    break

                #Look for meaningful title tags
                if outer.name in ["div", "p", "span", "h2", "h3", "h4"]:
                    text = outer.get_text(strip=True)
                    if not text:
                        continue
                    style = outer.get("style", "") if outer.has_attr("style") else ""
                    align = outer.get("align", "") if outer.has_attr("align") else ""

                    if not title_found and (
                        outer.name in ["h2", "h3", "h4"]
                        or "underline" in style
                        or "text-align:center" in style
                        or "center" in align
                        or "font-weight:700" in style
                    ):
                        table_title = text
                        title_found = True

                    elif not subtitle_found and (
                        "font-weight:600" in style
                        or "font-weight:700" in style
                        or outer.name in ["p", "span"]
                    ):
                        table_subtitle = text
                        subtitle_found = True

                if title_found and subtitle_found:
                    break

        # Next, extract column headers (if present)
        header_row = table.find("tr") #locates first instance of <tr> tag within the current <table> block
        if header_row:
            header_cells = header_row.find_all("td")
            current_column = 0
            for cell in header_cells:
                text = clean_value(cell.get_text(strip=True))
                if text:
                    colspan = int(cell.get("colspan", 1))  # Grabs the number of col span (e.g. <td colspan="3">Revenue</td> -> 3) Default colspan is 1 if not specified
                    headers.append(text) #appends the header text just once
                    current_column += colspan - 1  # Skip columns covered by this header

        # Then, extract rows of data, per row and appends into the table
        for row in table.find_all("tr")[1:]:  # Skip header row
            cells = row.find_all("td")
            row_data = []
            current_column = 0

            for cell in cells: 
                text = clean_value(cell.get_text(strip=True))
                colspan = int(cell.get("colspan", 1))  # Get colspan if available, else set it as 1 by default, e.g.  <td colspan="3">Revenue</td> returns 3. I.e. its just one <td> tag but the cell will visually fill 3 columns
                if text: #check that the cell has text
                    if not is_numeric(text):
                        row_data.append(text)
                    else:
                        # For numeric data, repeat it for the appropriate number of columns
                        for _ in range(colspan):
                            row_data.append(text) 
                    current_column += colspan

            if row_data:  # Only append rows that have meaningful data
                table_data.append(row_data) #note: here, table_data is a list, similar to row_data. hence appends just adds whatever we want to the back of the list

        # Lastly, try to convert to DataFrame **only if data is present**
        if table_data:
            # Ensure correct alignment of the header with data
            df = pd.DataFrame(table_data, columns=headers if headers else None) #table_data is list of lists (i.e. list of rows, each row is a list), but that's what pd.DataFrame() accepts.  Headers is a list of column names
            extracted_tables[f"Table_{table_count}"] = {
                "title": table_title,  # Include the table title
                "data": df             # Include the table data
            }
            table_count += 1  # Increment sequential table numbering

    return extracted_tables

def elevate_headers(df):
    """
    - Uses the first row as the actual header.
    - Drops the first row from data after elevating it.
    - Renames duplicate column names properly.
    - But can bolster with additional logic to check (1) if header already exists (so that we don't overwrite existing headers), and (2) check what we are actually promoting 
    """
    if df.empty:
        print("⚠️ WARNING: DataFrame is empty!")
        return df

    # Step 1: Promote first row as headers
    df.columns = df.iloc[0]  # Use first row as header. recall iloc is an index-based seleciton method that allows access to rows and columns using integer position. Usage: df.iloc[row_index, column_index]; df.iloc[0] selects first row; df.iloc[:,0] selects the first column
    df = df[1:].reset_index(drop=True)  # Drop first row and reset index

    #Step 2: Rename Duplicate Headers (e.g., "2024" → "2024_2")
    new_columns = []
    seen = {}

    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")  # Rename duplicate
        else:
            seen[col] = 1
            new_columns.append(col)

    df.columns = new_columns  # Apply new column names

    return df

def elevate_index(df):
    """
    Elevates the first column as the index **only if** it contains textual (non-numeric) data.
    Ensures that the index consists of meaningful labels rather than numeric values.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with the first column set as index (if applicable)
    """
    if df.empty or len(df.columns) == 0:
        print("⚠️ WARNING: DataFrame is empty or has no columns!")
        return df

    first_col = df.columns[0]  # Identify the first column

    # Check if the first column is text-based (i.e., not numeric)
    if df[first_col].dtype == 'object' or df[first_col].astype(str).str.isalpha().all(): #dtype == "object" usually indicates string value in pandas, whereas df[first_col].astype(str).str.isalpha().all() checks if all the values are made up of letters only (i.e. via the .str.isalpha()
        df = df.set_index(first_col)  # Elevate first column as index
        df.index.name = None  # Remove index label for clean output
        print(f"✅ Elevated '{first_col}' as index.")
    else:
        print(f"⚠️ First column '{first_col}' contains numeric data, not elevating.")

    return df

def get_cleaned_columns(df): 
    """
    - Keeps only one column per year 
    - Retains all "change" columns 
    - Keep all non-year columns 
    """

    if df.empty: 
        print("Warning: DataFrame is empty")
        return df 
    
    columns_to_keep = []
    seen_years = set()

    df.columns = [str(col) if col is not None else "Unnamed" for col in df.columns] #normalises column names to strings, replaces None with "Unnamed". recall df.column is a list (special one called pandas.index (which are the column index or the column labels)

    for col in df.columns: 
        #Keep all "change" columns 
        if "change" in col.lower(): 
            columns_to_keep.append(col)
            print(f"Keeping 'Change column: {col}")
            continue

        #extract year if its numeric using regular expressions
        match = re.search(r"(20\d{2})", col) #looks for 2020 to 2099. note: if replace with "\b(20\d{2})\b" -> then only captures standalone years. i.e. 2024 is a match but 2024_2 isn;t
        year = match.group(1) if match else None #extract the first match from the capturing group (20\d{2}). recall r prefix means raw string (e.g. \n isnt interpreted as escape characters). \d{2} matches exactly two digits
        #recall: capturing group in regex is anything wrapped in parentheses (...), tells the regex engine not only I want to match but save it so to refer to it later
        #group(0) returns full match, group(1) returns first group, ... group(n) returns the nth group. I.e. if our regex expression has more than one capturing group that we are trying to match
 
        if year: 
            if year not in seen_years: #only keep the first appearance of a given year
                seen_years.add(year) #but record it in the seen_years set
                columns_to_keep.append(col)
                print(f"Keeping year column: {col}")
            else: 
                print(f"Skipping duplicate year columns: {col}")


        else:
            if not re.search(r"\b(20\d{2})_\d+\b", col):
                print(f"Keeping non-year col: {col}")
                columns_to_keep.append(col)
    
    return columns_to_keep

def finalize_table(df, cleaned_columns):
    """
    - Drops columns that contain only None values.
    - Sets the first column as index (region names).
    """
    if df.empty:
        print("⚠️ WARNING: DataFrame is empty!")
        return df
    
    #Step 0: Print index before processing
    #print(f"\n🔎 [Before Processing] Index of {table_name}:")
    print(df.index)

    #Step 1: Ensure only relevant columns are kept, including the first column 
    #first_col = df.columns[0]
    #print(f"First column: {first_col}")
    #valid_columns = [first_col] + [col for col in df.columns if col in cleaned_columns or df[col].notna().any()]
    valid_columns = [col for col in df.columns if col in cleaned_columns or df[col].notna().any()] #we build a list of valid columns to keep, either if its in cleaned_columns list, or has at least one non-NaN value
    if not valid_columns: 
        print("All columns are empty, returning empty DF")
        return pd.DataFrame()

    # Step 2: Drop completely empty columns that are not in valid columns
    df = df[valid_columns]
    columns_to_drop = [col for col in df.columns if col not in cleaned_columns and df[col].isna().all()]
    df = df.drop(columns=columns_to_drop)

    return df

def remove_row_duplicates_correctly(df):
    """
    - Removes duplicate year columns row-by-row.
    - Keeps the first valid (non-None) occurrence of each value.
    - Ensures structure remains intact.
    - This works at the level of df 
    """
    if df.empty:
        print("⚠️ WARNING: DataFrame is empty!")
        return df

    cleaned_rows = [] #empty list to store the cleaned_rows

    for index, row in df.iterrows(): #going row by row 
        seen_values = set() #initialise empty set for seen values
        cleaned_row = [] #and empty list for clean rows 

        for value in row:
            if pd.isna(value) or value is None:  
                cleaned_row.append(None)  # Preserve None values -> later will have a compacting step that removes these
            elif value not in seen_values:  
                seen_values.add(value)
                cleaned_row.append(value)  # Keep first occurrence
            else:
                cleaned_row.append(None)  # Remove duplicate occurrences, replace with None 

        cleaned_rows.append(cleaned_row)

    # Convert back to DataFrame with the same structure
    df_cleaned = pd.DataFrame(cleaned_rows, columns=df.columns, index=df.index)

    return df_cleaned

def compact_table_no_padding(df):
    """
    - Removes None values row-by-row.
    - Fully compacts data by shifting all valid values left.
    - Does NOT preserve the original number of columns (removes excess empty columns).
    """
    if df.empty or df.columns is None or len(df.columns) == 0: 
        print("⚠️ WARNING: DataFrame have no valid columns!")
        return df

    #Debugging: check for headers 
    print(f"\n🔍 Debugging: Checking Columns for Table")
    print(f"Columns found: {df.columns.tolist()}")

    #Get clean headers 
    cleaned_columns = get_cleaned_columns(df)

    #Compact the data
    compacted_rows = []

    for index, row in df.iterrows():
        # Remove None values and only keep actual data
        row_values = [val for val in row if pd.notna(val)] #scan each row left to right, remove all blank cells and append it sequentially
        compacted_rows.append(row_values)
        
    # Convert back to DataFrame with cleaned structure (removes empty columns)
    df_compacted = pd.DataFrame(compacted_rows, index=df.index)  # No columns defined -> logic below handles that
    
    #Defining columns
    cleaned_columns = list(cleaned_columns)
    #Assigned clean headers dynamically 
    if len(df_compacted.columns) == len(cleaned_columns): 
        df_compacted.columns = cleaned_columns
    
    else: 
        print(f"⚠️ Column Mismatch: Compacting resulted in {len(df_compacted.columns)} columns, but cleaned headers are {len(cleaned_columns)}")
    
        if len(df_compacted.columns) < len(cleaned_columns): #if the compacted dataframe has less columns than that of clean, return the first few
            cleaned_columns = cleaned_columns[:len(df_compacted.columns)]  # Trim headers
        else:
            cleaned_columns += ["Unknown"] * (len(df_compacted.columns) - len(cleaned_columns))  # Fill extra columns
        df_compacted.columns = cleaned_columns
    return df_compacted

#Start to plot some tables 
import matplotlib
matplotlib.use('Agg') #uses non-interactive backend suitable for script-based image generation
import matplotlib.pyplot as plt 

def plot_data(filtered_tables, table_names, chart_types=None, years=None, save_dir=None): 
    """"
    Generalised function to plot data from filtered_tables 
    Parameters: 
    -filtered_tables (dict): Dictionary containing DataFrames 
    -table_name (list of str): name of the table to plot 
    -chart_type (list of str): "bar" or "line". Defaults to "bar"
    -years(list of str, optional): specific years to plot. Defaults to all available years

    Returns: 
    - Displays the chart
    """

    #Sets-up chart types and years (default to 'Line' and None)
    if chart_types is None: 
        chart_types = ['line'] * len(table_names) #default to line for all 

    if years is None: 
        years = [None] * len(table_names) #default to all years

    chart_paths = []
    
    for table_name, chart_type, selected_year in zip(table_names, chart_types, years): 
        #check if table name exists 
        if table_name not in filtered_tables: 
            print(f"Table '{table_name}' not found in retrieved form 10K tables")
            continue
    
        #Retrieve the right dataframe 
        df = filtered_tables[table_name]["data"]
        print(df.dtypes)
        print(df.head())

        df = df.replace(r"\(([\d,]+)\)", r"-\1", regex=True) #convert accounting style negatives before coercing to float
        df = df.apply(pd.to_numeric, errors='coerce')

        if df is None or df.empty: 
            print(f"Table {table_name} is empty!")
            continue 

        # 🔍 Step 0: Detect year columns and exclude anything with 'change'
        renamed_cols = {}
        year_pattern = re.compile(r"\b(20\d{2})(?:_\d+)?\b") #compiles a regular expression into a pattern object so that can run .search() on strings. \b: word boundary, + capturing group in ()
        seen_years = set()

        for col in df.columns:
            col_str = str(col)
            if (
                "change" in col_str.lower()
                or "%" in col_str
                or "percentage" in col_str.lower()
                or "percent" in col_str.lower()
            ):
                continue  # ❌ skip 'Change' columns

            match = year_pattern.search(col_str)
            if match:
                year = match.group(1)
                if year not in seen_years:
                    renamed_cols[col] = year
                    seen_years.add(year)

        # Rename those columns to clean year format (but leaves other str including change untouched)
        df = df.rename(columns=renamed_cols)

        # ✅ Step 1: Filter only valid year columns
        df_filtered = df.filter(regex=r"^20\d{2}$")

        # ✅ Step 2: If user selected specific years, restrict to those
        if selected_year:
            df_filtered = df_filtered.loc[:, df_filtered.columns.intersection(selected_year)]

        # ✅ Step 3: Safety check
        if df_filtered.empty:
            print(f"⚠️ No valid year columns found in {table_name}. Available columns: {df.columns.tolist()}")
            continue

        #Auto-detect year columns (exclude "Change columns")
        #df_filtered = df.filter(regex=r"20\d{2}")

        #if selected_year:
        #    df_filtered = df_filtered.loc[:, df_filtered.columns.intersection(selected_year)] #loc[row_selection, column_selection] used to select specific rows and columns in Pandas Dataframe. .columns.intersection(years) returns columns that exist in both df_with_years and years
        
        #if df_filtered.empty: 
            print(f"No valid year columns found in {table_name} after filtering")
        #    continue

        #Filter out rows with percentage inside 
        unwanted_keywords = ["percent", "percentage", "%"]
        df_filtered = df_filtered[~df.index.astype(str).str.lower().str.contains('|'.join(unwanted_keywords))]

        df_filtered = df_filtered.T #need this format (where each row corresponds to a year) to plot years on the x axis
        df_filtered = df_filtered.sort_index()
        df_filtered.index = df_filtered.index.astype(str)
        legend = df_filtered.columns.astype(str)

        ##**** Debugging print statements
        print("🧾 df_filtered before plotting:")
        print(df_filtered)
        print("🪪 Columns (legend):", df_filtered.columns)
        print("🎯 Index (x-axis):", df_filtered.index)
        ##******************************

        title = filtered_tables[table_name]["title"] #filtered tables is a dictionary where the keys = "table_name" (e.g. Table_7, Table_10) and the values are dictionaries with at least "title" and "data"
        #Plot 
        fig, ax = plt.subplots(figsize=(10,8))
        df_filtered.plot(kind=chart_type, ax=ax)
        ymin, ymax = df_filtered.min().min(), df_filtered.max().max()
        buffer = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - buffer, ymax + buffer)
        ax.set_title(title)
        ax.set_ylabel("Values")
        ax.set_xlabel("Year")
        ax.legend(legend, loc='upper left', bbox_to_anchor=(1.0, 0.5))
        ax.set_xticks(range(len(df_filtered.index)))
        ax.set_xticklabels(df_filtered.index, rotation=0)
        
        for line_idx, line in enumerate(ax.get_lines()):
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                try:
                    # Force y into float to handle numeric comparison
                    y_val = float(y)
                    if pd.notna(y_val):
                        base_offset = -10 if y_val < 0 else 5 + (line_idx * 10)
                        ax.annotate(f"{y_val:,.0f}",
                                    xy=(x, y_val),
                                    xytext=(0, base_offset),
                                    textcoords="offset points",
                                    ha="center",
                                    va = "bottom" if y_val < 0 else "top",
                                    fontsize=9,
                                    fontweight='normal',
                                    color=line.get_color())
                except Exception as e:
                    print(f"⚠️ Skipping annotation for y={y}: {e}")

        fig.tight_layout()

        if save_dir:
            filename = f"chart_{table_name}.png"
            full_path = os.path.join(save_dir, filename)
            fig.savefig(full_path)
            plt.close(fig)
            chart_paths.append(filename)
        else:
            plt.show()
    
    return chart_paths

def promote_index_if_row_blank(df: pd.DataFrame, current_title: str):
    """
    If the first row has a non-empty index label and all its values are NA or blank,
    promote the index as part of the title and remove the row.
    """
    if df.empty:
        print("⚠️ DataFrame is empty. Skipping promotion.")
        return df, current_title

    first_index_label = str(df.index[0]).strip()
    first_row = df.iloc[0]

    print(f"🔍 Checking if index label should be promoted...")
    print(f"🪪 First index label: '{first_index_label}'")
    print(f"📭 First row values: {list(first_row.values)}")

    if first_index_label and all((pd.isna(cell) or str(cell).strip() == "") for cell in first_row):
        new_title = f"{current_title} - {first_index_label}" if current_title else first_index_label
        print(f"✅ Promoting index label to title: {new_title}")
        df = df.drop(df.index[0])  # Drop the first row but preserve the index
        return df, new_title

    print("🚫 No promotion needed.")
    return df, current_title

def is_blank_header(h):
    if pd.isna(h): return True
    if isinstance(h, str):
        return h.strip() == "" or h.startswith("Unnamed")
    if isinstance(h, (int, float)):
        return True
    return False

def repair_broken_header(df: pd.DataFrame, offset=2):
    """
    Detects and fixes tables where headers are missing or default (like 0, 1, 2 or 'Unnamed').
    Uses the first row and its index label to reconstruct the header.
    Allows slight mismatches by padding or truncating the header row
    """
    if df.empty:
        print("⚠️ DataFrame is empty!")
        return df

    # Detect blank headers more robustly
    def is_blank_header(h):
        if pd.isna(h): return True
        if isinstance(h, str):
            return h.strip() == "" or h.startswith("Unnamed")
        if isinstance(h, (int, float)):
            return True
        return False
    
    def is_numeric_string(val):
        try:
            float(str(val).replace(",", "").replace("$", "").replace("%", "").strip())
            return True
        except ValueError:
            return False


    current_headers = list(df.columns)
    if all(is_blank_header(h) for h in current_headers):
        print("🔧 Attempting to repair broken header...")

        first_row = df.iloc[0]
        index_label = str(df.index[0]).strip()
        row_values = [str(v).strip() for v in first_row.tolist() if pd.notna(v) and str(v).strip() != ""]

        # Skip promotion if all values are numeric
        if row_values and all(is_numeric_string(v) for v in row_values):
            print("🚫 Skipping header promotion: First row is fully numeric.")
            return df


        potential_headers = [index_label] + row_values if index_label else row_values
        n_cols = df.shape[1]

        if len(potential_headers) == n_cols:
            print(f"✅ Promoting new header: {potential_headers}")
            df.columns = potential_headers
            df = df.drop(df.index[0])  # Drop the first row, preserve index
        else:
            print(f"⚠️ Header repair aborted: {len(potential_headers)} headers ≠ {df.shape[1]} columns")

    return df

def promote_index_if_row_values_blank(df: pd.DataFrame, current_title: str):
    """
    If the first index label is a non-empty string,
    and the corresponding row values are all blank/NaN,
    promote that index label as part of the title,
    but retain the rest of the index.
    """
    if df.empty:
        print("⚠️ DataFrame is empty. Skipping.")
        return df, current_title

    # Check that all index labels are non-empty strings
    if not all(isinstance(idx, str) and idx.strip() for idx in df.index):
        return df, current_title

    first_index = str(df.index[0]).strip()
    first_row_values = df.iloc[0].apply(lambda x: str(x).strip() if pd.notna(x) else "")

    print(f"🔍 Checking first index '{first_index}' with row values: {list(first_row_values)}")

    def is_blank_or_non_numeric(val):
        return val == "" or not any(char.isdigit() for char in val)

    if all(is_blank_or_non_numeric(val) for val in first_row_values):
        new_title = f"{current_title} - {first_index}" if current_title else first_index
        df = df.drop(df.index[0])  # Drops the row, retains original index
        print(f"✅ Promoted first index label to title: {new_title}")
        return df, new_title

    print("🚫 No promotion needed.")
    return df, current_title

def elevate_row_if_index_or_row_is_date(df: pd.DataFrame):
    """
    Elevates a combined header from the first index and first row 
    if both are date-like. Pads or truncates to match column count.
    """
    if df.empty:
        print("⚠️ DataFrame is empty.")
        return df

    def is_date_like(s):
        if not isinstance(s, str) or not s.strip():
            return False

        s = s.strip().lower()

        # Common month-year formats
        month_keywords = [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
        if any(m in s for m in month_keywords) and re.search(r"\d{4}", s):
            return True

        # Year-only formats (but must be between 1900–2100)
        if re.fullmatch(r"\d{4}", s):
            year = int(s)
            return 2021 <= year <= 2025

        return False


    # Get first row + index
    first_index = str(df.index[0]).strip()
    first_row = df.iloc[0].tolist()
    row_values = [str(v).strip() if pd.notna(v) else "" for v in first_row]

    # Check if index and all row values are date-like
    if is_date_like(first_index) and any(is_date_like(v) for v in row_values if v):
        combined = [first_index] + row_values
        n_cols = df.shape[1]

        # Pad or truncate to match
        if len(combined) < n_cols:
            combined += [""] * (n_cols - len(combined))
        elif len(combined) > n_cols:
            combined = combined[:n_cols]

        print(f"🪄 Promoting combined date row + index to header: {combined}")
        df.columns = combined
        df = df.drop(df.index[0])  # Drop header row only
        return df

    print("📭 No promotion: First index and row not date-like.")
    return df

def infer_title_from_index(df: pd.DataFrame, current_title: str = None) -> str:
    """
    Infers a fallback title for a table based on the longest string in the index.
    Useful when no table title was found during extraction.

    Parameters:
        df (pd.DataFrame): The table DataFrame.
        current_title (str): Existing title (if any).

    Returns:
        str: The final title, either the original or inferred from index.
    """
    if current_title and current_title.strip():
        return current_title  # Already have a title

    if df is None or df.empty or df.index is None:
        print("❌ Cannot infer title: DataFrame or index is empty.")
        return None

    index_candidates = df.index.astype(str).tolist()
    index_candidates = [s.strip() for s in index_candidates if s.strip()]

    if not index_candidates:
        print("❌ No valid index entries found to infer title.")
        return None

    # Optional: exclude short entries
    filtered_candidates = [s for s in index_candidates if len(s.split()) > 2 or len(s) > 10]

    if filtered_candidates:
        inferred_title = max(filtered_candidates, key=len)
    else:
        inferred_title = max(index_candidates, key=len)

    print(f"🧠 Fallback: Using longest index entry as title → '{inferred_title}'")
    return f"{inferred_title}**"

def plot_data_2(filtered_tables, table_names, chart_types=None, years=None, save_dir=None): 
    """"
    Generalised function to plot data from filtered_tables 
    Parameters: 
    - filtered_tables (dict): Dictionary containing DataFrames 
    - table_name (list of str): name of the table to plot 
    - chart_type (list of str): "bar" or "line". Defaults to "bar"
    - years (list of str, optional): specific years to plot. Defaults to all available years

    Returns: 
    - Displays the chart
    """
    if chart_types is None: 
        chart_types = ['line'] * len(table_names)

    if years is None: 
        years = [None] * len(table_names)

    chart_paths = []
    
    for table_name, chart_type, selected_year in zip(table_names, chart_types, years): 
        if table_name not in filtered_tables: 
            print(f"Table '{table_name}' not found in retrieved form 10K tables")
            continue
    
        df = filtered_tables[table_name]["data"]
        print(df.dtypes)
        print(df.head())

        df = df.apply(pd.to_numeric, errors='coerce')

        if df is None or df.empty: 
            print(f"Table {table_name} is empty!")
            continue 

        # ✅ Smarter year detection
        import re
        year_pattern = re.compile(r"(20\d{2})(?:_\d+)?$")
        year_columns = {}

        for col in df.columns:
            col_str = str(col)
            if "change" in col_str.lower():
                continue
            match = year_pattern.match(col_str)
            if match:
                normalized = match.group(1)
                if normalized not in year_columns:
                    year_columns[normalized] = col  # only keep the first occurrence

        sorted_years = sorted(year_columns.keys())
        sorted_columns = [year_columns[y] for y in sorted_years]
        df_filtered = df[sorted_columns]

        if selected_year:
            df_filtered = df_filtered.loc[:, df_filtered.columns.intersection(selected_year)]
        
        if df_filtered.empty: 
            print(f"No valid year columns found in {table_name} after filtering")
            continue

        df_filtered = df_filtered.T

        legend = df.index.astype(str)
        title = filtered_tables[table_name]["title"]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        df_filtered.plot(kind=chart_type, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Values")
        ax.set_xlabel("Year")
        ax.legend(legend)
        ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()

        if save_dir:
            filename = f"chart_{table_name}.png"
            full_path = os.path.join(save_dir, filename)
            fig.savefig(full_path)
            plt.close(fig)
            chart_paths.append(filename)
        else:
            plt.show()
    
    return chart_paths

def finalize_headers_with_padding(df):
    """
    Cleans and pads column headers using get_cleaned_columns.
    If extra columns are found at the end without names,
    tries to backfill them as decreasing years from the last valid year.
    If the last valid column is a "change" column, uses 'Change' instead.
    """
    if df.empty:
        print("⚠️ DataFrame is empty.")
        return df

    #cleaned_columns = get_cleaned_columns(df)
    cleaned_columns = list(df.columns)
    num_cols = df.shape[1]

    if len(cleaned_columns) < num_cols:
        print(f"🧼 Padding headers: {len(cleaned_columns)} found vs {num_cols} columns.")
        padding = [f"Unnamed_{i}" for i in range(num_cols - len(cleaned_columns))]
        cleaned_columns += padding

        import re
        year_pattern = re.compile(r"^(20\d{2})$")
        last_valid_col = None

        # Find the last valid column
        for col in reversed(cleaned_columns): #scanning the rightmost column
            if not col.startswith("Unnamed") and col.strip() != "": #skip those starting with unnamed and skips empty string or spaces, or None
                last_valid_col = col
                break #immediately exits the for loop so that we can proceed to the next if statement code block

        if last_valid_col: #assuming we found the last valid column header
            #First, see if the last valid column header is change, then we fill in the remaining unnamed columns with change
            if "change" in str(last_valid_col).lower():
                print(f"↩️ Backfilling with 'Change' based on last valid column: '{last_valid_col}'")
                for i in range(len(cleaned_columns)):
                    col = cleaned_columns[i]
                    if isinstance(col, str) and (col.strip() == "" or col.startswith("Unnamed")):
                        print(f"  ↪️ Fixing column {i}: was '{col}' → now 'Change'")
                        cleaned_columns[i] = "Change"
                    #if cleaned_columns[i].startswith("Unnamed") or cleaned_columns[i].strip() == "":
                    #    cleaned_columns[i] = "Change"
            #Else if its year, then backfil decreasing years 
            elif year_pattern.match(str(last_valid_col)):
                last_valid_year = int(year_pattern.match(str(last_valid_col)).group(1))
                print(f"↩️ Backfilling years from {last_valid_year}")
                for i in range(len(cleaned_columns)):
                    if cleaned_columns[i].startswith("Unnamed") or cleaned_columns[i].strip() == "":
                        last_valid_year -= 1
                        cleaned_columns[i] = str(last_valid_year)

    elif len(cleaned_columns) > num_cols:
        print(f"✂️ Trimming headers: {len(cleaned_columns)} > {num_cols} columns.")
        cleaned_columns = cleaned_columns[:num_cols]

    df.columns = cleaned_columns
    return df

if __name__ == "__main__": 
    from section_extractor import get_latest_10k_file
        
    file_path = get_latest_10k_file()
    if not file_path: 
        print("No 10-K HTML file found in downloads")
        sys.exit(1)
        
    print(f"Using file: {file_path}")

    #Step 1: Extract Raw Tables 
    raw_tables = extract_meaningful_tables(file_path)
    print(f"Extracted {len(raw_tables)} raw tables")

    #Step 2: Elevate headers/index, clean, finalize 
    filtered_tables = {}
    for table_name, table_info in raw_tables.items(): 
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

    
    for table_name, table_info in filtered_tables.items(): 
        df = table_info['data']
        title = table_info['title']

        if df is not None and not df.empty: 
            print(f"Finalising table format {table_name}: {title}")

            #Try repairing broken header 
            df = repair_broken_header(df) 
            #Check if there are dates in first row/index and promote to header 
            df = elevate_row_if_index_or_row_is_date(df)
            #Then, promoting index to title if its a blank row label (e.g. revenue, none none none)
            df, updated_title = promote_index_if_row_blank(df, title)
            #Then, promoting index if it is all strings 
            df, updated_title = promote_index_if_row_values_blank(df, updated_title)
            #Check if there are dates in first row/index and promote to header 
            df = elevate_row_if_index_or_row_is_date(df)
            #Final clean up of headers
            df = finalize_headers_with_padding(df)
            #Last fall back: infer title from index if all else fails 
            final_title = infer_title_from_index(df, updated_title)

            #Update table_info dictionary 
            filtered_tables[table_name]['data'] = df
            filtered_tables[table_name]['title'] = final_title

            print(df.head())
    
        else: 
            print(f"⚠️ Skipping {table_name}: Empty DataFrame")
    
    for table_name, table_info in filtered_tables.items(): 
        print(f"{table_name}: {table_info['title']}")

    # Create an "outputs" folder if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    output_path = os.path.join("outputs", "Cleaned_SEC_Tables.xlsx")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for name, table in filtered_tables.items():
            df = table["data"]
            if df is not None and not df.empty:
                sheet_name = name[:31]  # Excel sheet names must be ≤ 31 chars
                df.to_excel(writer, sheet_name=sheet_name, index=True)

    print(f"📁 All tables saved to: {output_path}")
    
    table_names = ['Table_7', 'Table_8']
    chart_paths = plot_data(filtered_tables, table_names, save_dir="static")
    print(chart_paths)


        
