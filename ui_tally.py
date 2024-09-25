import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
import re

# Function to add a serial number column to DataFrame
def add_serial_number_column(df):
    # Remove duplicates of 'sr. no.' if present
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Add serial number column
    if 'sr. no.' not in df.columns:
        df.insert(0, 'sr. no.', range(1, len(df) + 1))
    
    return df

# Function to display DataFrame with additional statistics
def display_dataframe_with_stats(df, name, sum_column):
    st.header(name)
    st.dataframe(add_serial_number_column(df))
    
    row_count = len(df)
    total_sum = df[sum_column].sum() if sum_column in df.columns else 0
    
    st.write(f"**{name} - Total Rows:** {row_count}")
    st.write(f"**{name} - Sum of {sum_column}:** {total_sum:.2f}")


#Adds an empty line after every occurrence of the target line in the input content
def add_empty_line(input_content, target_line):
    output = StringIO()
    # Split the input content into lines based on newline characters
    for line in input_content.split('\n'):
        output.write(line + '\n')
        # If the current line matches the target line (ignoring leading/trailing spaces)
        if line.strip() == target_line.strip():
            output.write('\n')
    return output.getvalue()


#Adds a single line break after the line containing 'Sr. No.' in the section of the content that follows the header '^PART-I - Details of Tax Deducted at Source^'.

def add_line_breaker_to_content(content):
    sections = content.split('^PART-I - Details of Tax Deducted at Source^')
    
    if len(sections) < 2:
        raise ValueError("Expected header not found in the file")

    header_section = sections[0]
    data_section = sections[1]

    lines = data_section.strip().split('\n')
    modified_lines = []
    header_found = False

    for line in lines:
        if not header_found and "Sr. No." in line:
            modified_lines.append(line)
            modified_lines.append(' ' * 1)
            header_found = True
        else:
            modified_lines.append(line)

    # Reassemble the content by joining the header section with the modified data section

    modified_content = header_section + '^PART-I - Details of Tax Deducted at Source^' + '\n'.join(modified_lines)
    return modified_content

def read_data_from_content(content):
    """Extracts the header and data from the TDS file content."""
    
    # Split content at the section with the details of tax deducted
    sections = content.split('^PART-I - Details of Tax Deducted at Source^')[1].split('\n\n')

    all_data = []
    header = None

    # Iterate through each section to extract data
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue

        # Extract deductor information from the first line of each section
        deductor_info = lines[0].split('^')
        if len(deductor_info) < 3:
            continue

        deductor_number = deductor_info[0]
        deductor_name = deductor_info[1]
        deductor_tan = deductor_info[2]

        # Process the remaining lines, looking for the header and relevant data
        for line in lines[1:]:
            if line.strip() == '':
                continue
            
            # Split columns by '^' and remove empty entries
            cols = [col.strip() for col in line.split('^') if col.strip()]
            
            # Identify and store the header
            if not header and "Sr. No." in cols:
                header = cols
            # Append data rows that match the length of the header
            elif header and cols and cols[0].isdigit() and len(cols) == len(header):
                all_data.append([deductor_number, deductor_name, deductor_tan] + cols)

    # Raise an error if the header is not found
    if not header:
        raise ValueError("Header not found in the file")

    # Combine deductor information and header to form the full header
    full_header = ['Deductor Number', 'Name of Deductor', 'TAN of Deductor'] + header
    
    return full_header, all_data


def create_tds_dataframe(header, data):
    df = pd.DataFrame(data, columns=header)
    numeric_columns = ['Amount Paid / Credited(Rs.)', 'Tax Deducted(Rs.)', 'TDS Deposited(Rs.)']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def process_tds_file(uploaded_file):
    """Reads and processes the TDS file to extract data and create a DataFrame."""
    
    # Decode the uploaded file content from bytes to string
    content = uploaded_file.getvalue().decode("utf-8")
    
    # Identify the line that contains the header of the TDS data
    target_line = "Sr. No.^Name of Deductor^TAN of Deductor^^^^^Total Amount Paid / Credited(Rs.)^Total Tax Deducted(Rs.)^Total TDS Deposited(Rs.)"
    
    # Add an empty line after the header for correct parsing
    content_with_empty_line = add_empty_line(content, target_line)
    
    # Modify content to ensure proper line breaks for processing
    modified_content = add_line_breaker_to_content(content_with_empty_line)
    
    # Separate the header and data from the modified content
    header, data = read_data_from_content(modified_content)
    
    # Create a DataFrame using the header and data
    df = create_tds_dataframe(header, data)
    
    # Drop unnecessary columns such as 'Deductor Number' and 'Sr. No.'
    df = df.drop(columns=['Deductor Number', 'Sr. No.'], errors='ignore')

    # Group data by 'Name of Deductor' and 'TAN of Deductor' and aggregate the TDS-related amounts
    aggregated_tds_df = df.groupby(['Name of Deductor', 'TAN of Deductor']).agg({
        'Amount Paid / Credited(Rs.)': 'sum',
        'Tax Deducted(Rs.)': 'sum',
        'TDS Deposited(Rs.)': 'sum'
    }).reset_index()
    
    # Return the original and aggregated DataFrames
    return df, aggregated_tds_df


# Function to process the Zoho file with dynamic column selection
def preprocess_zoho_file_with_selection(file):
    df = pd.read_excel(file, header=None)

    # Function to detect the start of the table by checking for a row that contains all non-null or non-NaN values
    def find_header_row(data):
        for i, row in data.iterrows():
            if row.notna().sum() > (len(row) // 2):  # Assuming header row has more than half non-NaN values
                return i
        return None

    # Find the header row dynamically
    header_row = find_header_row(df)
    
    # Extract table data starting from the header row
    if header_row is not None:
        table_df = pd.read_excel(file, header=header_row)
        table_df = table_df.dropna(how='all')  # Drop rows that are completely empty

        if 'Particulars' in table_df.columns:
            table_df = table_df[table_df['Particulars'].notna()]  # Remove rows where 'Particulars' is NaN
            table_df = table_df[table_df['Particulars'].astype(str).str.strip() != '']  # Remove rows where 'Particulars' is empty or whitespace

        # Adding sub heading for column selection....
        st.sidebar.title("File Uploads")

        # Let user select columns
        col1 = st.sidebar.selectbox('Select the column for "Name of Deductor"', table_df.columns)
        col2 = st.sidebar.selectbox('Select the column for "TDS of the Current Financial Year"', table_df.columns)
        col3 = st.sidebar.selectbox('Select the column for "Total Amount"', table_df.columns)

        # Return selected columns as DataFrame
        selected_df = table_df[[col1, col2, col3]].copy()
        selected_df.columns = ['name of the deductor', 'tds of the current fin. year', 'total amount']

        # Convert 'name of the deductor' column values to uppercase for uniformity
        selected_df['name of the deductor'] = selected_df['name of the deductor'].str.upper()
        # Convert 'tds of the current fin. year' column to numeric, handling any conversion errors
        selected_df['tds of the current fin. year'] = pd.to_numeric(selected_df['tds of the current fin. year'], errors='coerce')
        # selected_df['total amount'] = pd.to_numeric(selected_df['total amount'], errors='coerce')

        # Remove any rows where 'tds of the current fin. year' could not be converted to a valid number (i.e., NaN values)
        selected_df = selected_df.dropna(subset=['tds of the current fin. year'])

        # Remove the last row (if it's assumed to be a total or irrelevant)
        selected_df = selected_df.iloc[:-1]

        # Group by 'name of the deductor' and sum 'tds of the current fin. year'
        aggregated_df = selected_df.groupby('name of the deductor', as_index=False)['tds of the current fin. year'].sum()

        return aggregated_df, selected_df

    else:
        st.error("No valid header row found in the Zoho file.")
        return None, None



def compare_dataframes(aggregated_tds_df, aggregated_zoho_df):
    """Compares TDS and Zoho DataFrames and returns a new DataFrame with exact matching entries."""
    
    # Perform an inner join between TDS DataFrame and Zoho DataFrame
    # on 'Name of Deductor' and 'TDS Deposited(Rs.)' from TDS DataFrame
    # and 'name of the deductor' and 'tds of the current fin. year' from Zoho DataFrame
    matching_df = pd.merge(
        aggregated_tds_df,  # TDS DataFrame
        aggregated_zoho_df[['name of the deductor', 'tds of the current fin. year']],  # Zoho DataFrame with relevant columns
        left_on=['Name of Deductor', 'TDS Deposited(Rs.)'],  # Columns from TDS DataFrame to match on
        right_on=['name of the deductor', 'tds of the current fin. year'],  # Columns from Zoho DataFrame to match on
        how='inner'  # Inner join ensures only exact matches are retained
    )
    
    # Rename the columns from Zoho DataFrame for clarity
    matching_df = matching_df.rename(
        columns={
            'name of the deductor': 'Name of Deductor (Zoho)',  # Rename for clarity in the result
            'tds of the current fin. year': 'TDS of the Current Fin. Year (Zoho)'  # Rename to indicate Zoho data source
        }
    )
    
    # Return the DataFrame containing only exact matches between TDS and Zoho data
    return matching_df


def compare_with_tolerance(aggregated_tds_df, aggregated_zoho_df, exact_matches, tolerance=10):
    """Compares TDS and Zoho DataFrames within a tolerance and excludes exact matches."""
    # Merge with tolerance and exclude exact matches
    tolerance_df = pd.merge(
        aggregated_tds_df,
        aggregated_zoho_df[['name of the deductor', 'tds of the current fin. year']],
        left_on='Name of Deductor',
        right_on='name of the deductor',
        how='inner'
    )

    # Filter within tolerance and exclude exact matches
    tolerance_df = tolerance_df[
        (abs(tolerance_df['TDS Deposited(Rs.)'] - tolerance_df['tds of the current fin. year']) <= tolerance) &
        (~tolerance_df['Name of Deductor'].isin(exact_matches['Name of Deductor'])) &
        (~tolerance_df['TDS Deposited(Rs.)'].isin(exact_matches['TDS Deposited(Rs.)']))
    ]

    tolerance_df = tolerance_df.rename(
        columns={
            'name of the deductor': 'Name of Deductor (Zoho)',
            'tds of the current fin. year': 'TDS of the Current Fin. Year (Zoho)'
        }
    )
    return tolerance_df

def get_unmatched_entries(aggregated_tds_df, aggregated_zoho_df, exact_matching_df, tolerance_matching_df):
    """Finds unmatched entries between TDS and Zoho DataFrames, excluding exact and tolerance matches."""
    # Combine exact and tolerance matches to exclude from unmatched
    combined_matches = pd.concat([exact_matching_df[['Name of Deductor', 'TDS Deposited(Rs.)']],
                                  tolerance_matching_df[['Name of Deductor', 'TDS Deposited(Rs.)']]]).drop_duplicates()

    # Unmatched in TDS
    unmatched_tds = aggregated_tds_df[~aggregated_tds_df.set_index(['Name of Deductor', 'TDS Deposited(Rs.)']).index.isin(combined_matches.set_index(['Name of Deductor', 'TDS Deposited(Rs.)']).index)]

    # Combine exact and tolerance matches to exclude from unmatched
    combined_matches_zoho = pd.concat([exact_matching_df[['Name of Deductor (Zoho)', 'TDS of the Current Fin. Year (Zoho)']],
                                       tolerance_matching_df[['Name of Deductor (Zoho)', 'TDS of the Current Fin. Year (Zoho)']]]).drop_duplicates()

    # Unmatched in Zoho
    unmatched_zoho = aggregated_zoho_df[~aggregated_zoho_df.set_index(['name of the deductor', 'tds of the current fin. year']).index.isin(combined_matches_zoho.set_index(['Name of Deductor (Zoho)', 'TDS of the Current Fin. Year (Zoho)']).index)]

    return unmatched_tds, unmatched_zoho

def get_individual_unmatched_entries(df, unmatched_df, key_col, sum_col):
    # """Gets individual entries for unmatched deductors."""
    """Gets individual entries for unmatched deductors."""
    unmatched_deductors = unmatched_df[key_col].unique()
    individual_unmatched_df = df[df[key_col].isin(unmatched_deductors)]
    return individual_unmatched_df

@st.cache_data
def convert_df_to_excel(df):
    # """Converts DataFrame to Excel format for download."""
    """Converts DataFrame to Excel format for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def match_individual_entries(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Matches individual entries between TDS and Zoho DataFrames and returns a new DataFrame with matched entries."""
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key and sum columns match
            if (row_tds[key_col_tds] == row_zoho[key_col_zoho] and 
                row_tds[sum_col_tds] == row_zoho[sum_col_zoho]):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    return matched_df

def match_individual_entries_with_tolerance(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho, tolerance):
    """Matches individual entries between TDS and Zoho DataFrames within a tolerance and returns a new DataFrame with matched entries."""
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame within the tolerance range
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key columns match and sum columns are within the tolerance range
            if (row_tds[key_col_tds] == row_zoho[key_col_zoho] and 
                abs(row_tds[sum_col_tds] - row_zoho[sum_col_zoho]) <= tolerance):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    return matched_df

def get_remaining_unmatched_entries(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing only the exact matched ones."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho

def get_remaining_unmatched_entries_with_tolerance(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing only the matched ones within the tolerance range."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho

def extract_first_three_words(name):
    """Extracts the first three words from the given string."""
    if isinstance(name, str):
        return ' '.join(name.split()[:3])
    return ''

def match_individual_entries_based_on_three_words(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Matches individual entries between TDS and Zoho DataFrames based on the first three words in the deductor name."""
    
    # Extract the first three words from the deductor names
    unmatched_tds['Three Words Deductor (TDS)'] = unmatched_tds[key_col_tds].apply(extract_first_three_words)
    unmatched_zoho['Three Words Deductor (Zoho)'] = unmatched_zoho[key_col_zoho].apply(extract_first_three_words)
    
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key columns match on first three words and sum columns match
            if (row_tds['Three Words Deductor (TDS)'] == row_zoho['Three Words Deductor (Zoho)'] and 
                row_tds[sum_col_tds] == row_zoho[sum_col_zoho]):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    
    # Drop the temporary columns used for matching
    unmatched_tds.drop(columns=['Three Words Deductor (TDS)'], inplace=True)
    unmatched_zoho.drop(columns=['Three Words Deductor (Zoho)'], inplace=True)
    
    return matched_df

def get_remaining_unmatched_entries_after_three_words(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing the matched ones based on the first three words in the deductor name."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho

def match_individual_entries_with_tolerance_based_on_three_words(unmatched_tds, unmatched_zoho, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho, tolerance):
    """Matches individual entries between TDS and Zoho DataFrames within a tolerance based on the first three words in the deductor name."""
    
    # Extract the first three words from the deductor names
    unmatched_tds['Three Words Deductor (TDS)'] = unmatched_tds[key_col_tds].apply(extract_first_three_words)
    unmatched_zoho['Three Words Deductor (Zoho)'] = unmatched_zoho[key_col_zoho].apply(extract_first_three_words)
    
    matched_indices_tds = set()
    matched_indices_zoho = set()
    
    matched_entries = []

    # Iterate over TDS DataFrame
    for idx_tds, row_tds in unmatched_tds.iterrows():
        # Skip if TDS entry already matched
        if idx_tds in matched_indices_tds:
            continue
        
        # Find a matching entry in Zoho DataFrame within the tolerance range
        for idx_zoho, row_zoho in unmatched_zoho.iterrows():
            # Skip if Zoho entry already matched
            if idx_zoho in matched_indices_zoho:
                continue
            
            # Check if both key columns match on first three words and sum columns are within the tolerance range
            if (row_tds['Three Words Deductor (TDS)'] == row_zoho['Three Words Deductor (Zoho)'] and 
                abs(row_tds[sum_col_tds] - row_zoho[sum_col_zoho]) <= tolerance):
                
                # Add to matched entries list
                matched_entries.append({
                    key_col_tds: row_tds[key_col_tds],
                    sum_col_tds: row_tds[sum_col_tds],
                    key_col_zoho: row_zoho[key_col_zoho],
                    sum_col_zoho: row_zoho[sum_col_zoho]
                })
                
                # Mark these indices as matched
                matched_indices_tds.add(idx_tds)
                matched_indices_zoho.add(idx_zoho)
                
                # Break after finding the first match
                break

    # Convert matched entries to DataFrame
    matched_df = pd.DataFrame(matched_entries)
    
    # Drop the temporary columns used for matching
    unmatched_tds.drop(columns=['Three Words Deductor (TDS)'], inplace=True)
    unmatched_zoho.drop(columns=['Three Words Deductor (Zoho)'], inplace=True)
    
    return matched_df

def get_remaining_unmatched_entries_after_tolerance_three_words(individual_unmatched_tds, individual_unmatched_zoho, matched_df, key_col_tds, key_col_zoho, sum_col_tds, sum_col_zoho):
    """Finds the remaining unmatched individual entries after removing the matched ones based on tolerance of the first three words in the deductor name."""

    # Create copies of the unmatched DataFrames to avoid modifying the originals
    remaining_unmatched_tds = individual_unmatched_tds.copy()
    remaining_unmatched_zoho = individual_unmatched_zoho.copy()

    # Convert matched_df to a list of dictionaries for efficient row-wise processing
    matched_rows = matched_df.to_dict('records')

    # Iterate through each matched row to remove only the exact matches
    for matched_row in matched_rows:
        # Find the first exact match in the TDS DataFrame
        tds_matches = remaining_unmatched_tds[
            (remaining_unmatched_tds[key_col_tds] == matched_row[key_col_tds]) &
            (remaining_unmatched_tds[sum_col_tds] == matched_row[sum_col_tds])
        ]

        # Remove only the first matched row in TDS if it exists
        if not tds_matches.empty:
            first_match_idx = tds_matches.index[0]
            remaining_unmatched_tds = remaining_unmatched_tds.drop(index=first_match_idx)

        # Find the first exact match in the Zoho DataFrame
        zoho_matches = remaining_unmatched_zoho[
            (remaining_unmatched_zoho[key_col_zoho] == matched_row[key_col_zoho]) &
            (remaining_unmatched_zoho[sum_col_zoho] == matched_row[sum_col_zoho])
        ]

        # Remove only the first matched row in Zoho if it exists
        if not zoho_matches.empty:
            first_match_idx = zoho_matches.index[0]
            remaining_unmatched_zoho = remaining_unmatched_zoho.drop(index=first_match_idx)

    return remaining_unmatched_tds, remaining_unmatched_zoho



def format_indian_currency(amount):
    """
    Formats a number or string as Indian currency without using locale settings.
    Handles inputs that may already be partially formatted.
    """
    # If input is a string, remove existing commas and currency symbol
    if isinstance(amount, str):
        amount = amount.replace(',', '').replace('₹', '').strip()
    
    # Convert the number to a string with two decimal places
    formatted_amount = "{:.2f}".format(float(amount))
    
    # Split the formatted amount into integer and decimal parts
    parts = formatted_amount.split(".")
    integer_part = parts[0]
    decimal_part = parts[1]
    
    # Format the integer part for Indian currency style
    if len(integer_part) > 3:
        # Get the last three digits for the hundreds place
        last_three = integer_part[-3:]
        # Format the remaining part with commas after every two digits from right to left
        remaining_digits = integer_part[:-3]
        formatted_remaining = ""
        for i, digit in enumerate(remaining_digits[::-1]):
            if i > 0 and i % 2 == 0:
                formatted_remaining = ',' + formatted_remaining
            formatted_remaining = digit + formatted_remaining
        # Combine the two parts with a comma
        integer_part = formatted_remaining + ',' + last_three if formatted_remaining else last_three
    
    # Reassemble the formatted amount
    formatted_amount = integer_part + "." + decimal_part
    
    return f"₹{formatted_amount}"





def create_summary_tables(final_matched_df, remaining_unmatched_tds, remaining_unmatched_zoho, selected_columns, df):
    # Calculate total amounts
    total_zoho = selected_columns["tds of the current fin. year"].sum()
    total_26as = df["TDS Deposited(Rs.)"].sum()
    
    # Calculate amounts for each category
    matched_amount = final_matched_df["TDS Deposited(Rs.)"].sum()
    unmatched_26as_amount = remaining_unmatched_tds["TDS Deposited(Rs.)"].sum()
    unmatched_zoho_amount = remaining_unmatched_zoho["tds of the current fin. year"].sum()
    
    # Calculate counts
    total_count_26as = len(df)
    total_count_zoho = len(selected_columns)
    matched_count = len(final_matched_df)
    unmatched_26as_count = len(remaining_unmatched_tds)
    unmatched_zoho_count = len(remaining_unmatched_zoho)
    
    # Calculate percentages for amounts
    matched_percent_zoho = (matched_amount / total_zoho) * 100 if total_zoho != 0 else 0
    matched_percent_26as = (matched_amount / total_26as) * 100 if total_26as != 0 else 0
    unmatched_26as_percent = (unmatched_26as_amount / total_26as) * 100 if total_26as != 0 else 0
    unmatched_zoho_percent = (unmatched_zoho_amount / total_zoho) * 100 if total_zoho != 0 else 0

    # Calculate percentages for counts
    matched_count_percent_26as = (matched_count / total_count_26as) * 100 if total_count_26as != 0 else 0
    matched_count_percent_zoho = (matched_count / total_count_zoho) * 100 if total_count_zoho != 0 else 0
    unmatched_26as_count_percent = (unmatched_26as_count / total_count_26as) * 100 if total_count_26as != 0 else 0
    unmatched_zoho_count_percent = (unmatched_zoho_count / total_count_zoho) * 100 if total_count_zoho != 0 else 0

    # Create count summary table
    count_summary_data = {
        "Category": ["Original Count", "Matched Count", "Unmatched Count"],
        "26AS": [
            total_count_26as,
            matched_count,
            unmatched_26as_count
        ],
        "26AS %": [
            "100%",
            f"{matched_count_percent_26as:.2f}%",
            f"{unmatched_26as_count_percent:.2f}%"
        ],
        "Zoho": [
            total_count_zoho,
            matched_count,
            unmatched_zoho_count
        ],
        "Zoho %": [
            "100%",
            f"{matched_count_percent_zoho:.2f}%",
            f"{unmatched_zoho_count_percent:.2f}%"
        ]
    }
    
    count_summary_df = pd.DataFrame(count_summary_data)
    count_summary_df = count_summary_df.set_index("Category")
    
    # Create amount summary table
    amount_summary_data = {
        "Category": ["Original Amount", "Matched Amount", "Unmatched Amount"],
        "26AS": [
            format_indian_currency(total_26as),
            format_indian_currency(matched_amount),
            format_indian_currency(unmatched_26as_amount)
        ],
        "26AS %": [
            "100%",
            f"{matched_percent_26as:.2f}%",
            f"{unmatched_26as_percent:.2f}%"
        ],
        "Zoho": [
            format_indian_currency(total_zoho),
            format_indian_currency(matched_amount),
            format_indian_currency(unmatched_zoho_amount)
        ],
        "Zoho %": [
            "100%",
            f"{matched_percent_zoho:.2f}%",
            f"{unmatched_zoho_percent:.2f}%"
        ]
    }
    
    amount_summary_df = pd.DataFrame(amount_summary_data)
    amount_summary_df = amount_summary_df.set_index("Category")
    
    return count_summary_df, amount_summary_df

def display_summary_tables(count_summary_df, amount_summary_df):
    st.header("Summary Tables")
    
    st.subheader("Count of Entries")
    st.table(count_summary_df)
    
    st.subheader("Amount of Entries")
    st.table(amount_summary_df)
    
    # Create download buttons for summary tables
    count_summary_excel = convert_df_to_excel(count_summary_df.reset_index())
    st.download_button("Download Count Summary Table", count_summary_excel, "Count_Summary_Table.xlsx")
    
    amount_summary_excel = convert_df_to_excel(amount_summary_df.reset_index())
    st.download_button("Download Amount Summary Table", amount_summary_excel, "Amount_Summary_Table.xlsx")


# # Initialize session state variables
# if 'aggregated_tds_df' not in st.session_state:
#     st.session_state.aggregated_tds_df = None
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'aggregated_zoho_df' not in st.session_state:
#     st.session_state.aggregated_zoho_df = None
# if 'selected_columns' not in st.session_state:
#     st.session_state.selected_columns = None
# if 'final_matched_df' not in st.session_state:
#     st.session_state.final_matched_df = None
# if 'remaining_unmatched_tds' not in st.session_state:
#     st.session_state.remaining_unmatched_tds = pd.DataFrame()  # Initialize with empty DataFrame
# if 'remaining_unmatched_zoho' not in st.session_state:
#     st.session_state.remaining_unmatched_zoho = pd.DataFrame()  # Initialize with empty DataFrame
# Initialize session state variables
default_dataframes = ['remaining_unmatched_tds', 'remaining_unmatched_zoho']

for var in ['df', 'aggregated_tds_df', 'aggregated_zoho_df', 'selected_columns', 'final_matched_df', 'final_matched_df_selected']:
    if var not in st.session_state:
        st.session_state[var] = None

for df_var in default_dataframes:
    if df_var not in st.session_state:
        st.session_state[df_var] = pd.DataFrame()




# Main function for Streamlit app
def main():
    st.title("TDS Reconciliation Tool")
    st.sidebar.title("File Uploads")

    # About the Tool
    st.markdown("""
    ### About the Tool
    **Reconciliation Purpose**  
    This tool is designed to reconcile TDS data between your books of accounts (TDS Ledger) and Form 26AS.

    **Objective**  
    The goal is to minimize unmatched TDS entries as much as possible. The tool will provide reconciliation items that can be matched one-on-one with the TDS ledger.

    **Current Functionality**  
    At present, we are matching data based on three types:
    1. **Totality Basis** - Summarizes TDS data on a totality basis.   
    2. **Individual Basis** - Matches data on an individual record basis.   
    3. **First Three Words of Deductor** - Matches based on the first three words of the deductor's name. 
                
    Additionally, the tool matches entries in two patterns:
    1. **Exact Matches** - Matches where the entries are exactly the same.    
    2. **Matches by Tolerance** - Matches where there is a tolerance of ±10%. 

    **Output**  
    The tool will generate the following columns in sequential order:

    1. Deductor Number  
    2. Name of Deductor  
    3. TAN of Deductor  
    4. Sr. No.  
    5. Section  
    6. Transaction Date  
    7. Status of Booking  
    8. Date of Booking  
    9. Remarks  
    10. Total Amount Paid / Credited (Rs.)  
    11. Total Tax Deducted (Rs.)  
    12. Total TDS Deposited (Rs.)

    *Some columns may remain empty as the data is retrieved on a totality basis.*

    Kindly note this tool is currently under development. Please review the results carefully before relying on them.
    """)


    # Initialize session state variables
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'aggregated_tds_df' not in st.session_state:
        st.session_state.aggregated_tds_df = None
    if 'aggregated_zoho_df' not in st.session_state:
        st.session_state.aggregated_zoho_df = None
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = None
    if 'final_matched_df' not in st.session_state:
        st.session_state.final_matched_df = None
    if 'remaining_unmatched_tds' not in st.session_state:
        st.session_state.remaining_unmatched_tds = pd.DataFrame()
    if 'remaining_unmatched_zoho' not in st.session_state:
        st.session_state.remaining_unmatched_zoho = pd.DataFrame()
    if 'final_matched_df_selected' not in st.session_state:
        st.session_state.final_matched_df_selected = pd.DataFrame()

    # File uploaders
    uploaded_file = st.sidebar.file_uploader("Upload a Text File (26AS TDS File)", type=["txt"])
    zoho_file = st.sidebar.file_uploader("Upload Tally Excel File", type=["xlsx", "xls"])

    # If Zoho file uploaded, allow user to select columns before submitting
    if zoho_file is not None:
        aggregated_zoho_df, selected_columns = preprocess_zoho_file_with_selection(zoho_file)
        if aggregated_zoho_df is not None:
            st.write("Selected Zoho Data:")
            st.dataframe(selected_columns)
            # Save to session state for further processing after submission
            st.session_state.aggregated_zoho_df = aggregated_zoho_df
            st.session_state.selected_columns = selected_columns

    # Button to start the process
    if st.sidebar.button("Submit"):
        if uploaded_file is not None:
            try:
                # Process TDS File
                df, aggregated_tds_df = process_tds_file(uploaded_file)
                display_dataframe_with_stats(df, "I. 26AS Extracted Data", "TDS Deposited(Rs.)")
                display_dataframe_with_stats(aggregated_tds_df, "II. Aggregated Totals by Deductor (26AS TDS)", "TDS Deposited(Rs.)")

                # Save DataFrames to session state
                st.session_state.df = df
                st.session_state.aggregated_tds_df = aggregated_tds_df

                # Proceed with Tally file processing
                if st.session_state.aggregated_zoho_df is not None:
                    display_dataframe_with_stats(st.session_state.selected_columns, "III. Tally Extracted Data (Individual Records)", "tds of the current fin. year")
                    display_dataframe_with_stats(st.session_state.aggregated_zoho_df, "IV. Aggregated Totals by Deductor (Tally)", "tds of the current fin. year")
                    
                    # Save Zoho DataFrames to session state
                    st.session_state.aggregated_zoho_df = aggregated_zoho_df
                    st.session_state.selected_columns = selected_columns

                    # Perform exact match comparison
                    exact_matching_df = compare_dataframes(aggregated_tds_df, aggregated_zoho_df)
                    display_dataframe_with_stats(exact_matching_df, "V. Aggregate Matching of TDS and Tally", "TDS Deposited(Rs.)")

                    # Perform tolerance match comparison
                    tolerance_matching_df = compare_with_tolerance(aggregated_tds_df, aggregated_zoho_df, exact_matching_df, tolerance=10)
                    display_dataframe_with_stats(tolerance_matching_df, "VI. Tolerance Matching of TDS and Tally (Within ±10)", "TDS Deposited(Rs.)")

                    # Get unmatched entries
                    unmatched_tds, unmatched_zoho = get_unmatched_entries(aggregated_tds_df, aggregated_zoho_df, exact_matching_df, tolerance_matching_df)
                    display_dataframe_with_stats(unmatched_tds, "VII. Unmatched Entries in TDS", "TDS Deposited(Rs.)")
                    display_dataframe_with_stats(unmatched_zoho, "VIII. Unmatched Entries in Tally", "tds of the current fin. year")

                    # Display individual unmatched deductor data
                    individual_unmatched_tds = get_individual_unmatched_entries(df, unmatched_tds, 'Name of Deductor', 'TDS Deposited(Rs.)')
                    display_dataframe_with_stats(individual_unmatched_tds, "IX. Individual Unmatched Deductors in TDS", "TDS Deposited(Rs.)")

                    individual_unmatched_zoho = get_individual_unmatched_entries(selected_columns, unmatched_zoho, 'name of the deductor', 'tds of the current fin. year')
                    display_dataframe_with_stats(individual_unmatched_zoho, "X. Individual Unmatched Deductors in Zoho", "tds of the current fin. year")

                    # Perform matching on individual unmatched entries
                    individual_matched_df = match_individual_entries(
                        individual_unmatched_tds,
                        individual_unmatched_zoho,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year'
                    )
                    display_dataframe_with_stats(individual_matched_df, "XI. Matched Individual Unmatched Entries in TDS and Tally", "TDS Deposited(Rs.)")

                    # Get remaining unmatched individual entries after removing matched ones
                    remaining_unmatched_tds, remaining_unmatched_zoho = get_remaining_unmatched_entries(
                        individual_unmatched_tds,
                        individual_unmatched_zoho,
                        individual_matched_df,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year'
                    )

                    # Display remaining unmatched individual entries
                    display_dataframe_with_stats(remaining_unmatched_tds, "XII. Remaining Unmatched Individual Entries in TDS", "TDS Deposited(Rs.)")
                    display_dataframe_with_stats(remaining_unmatched_zoho, "XIII. Remaining Unmatched Individual Entries in Tally", "tds of the current fin. year")

                    # Perform tolerance matching on individual unmatched entries
                    individual_tolerance_matched_df = match_individual_entries_with_tolerance(
                        remaining_unmatched_tds,
                        remaining_unmatched_zoho,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year',
                        tolerance=10
                    )
                    display_dataframe_with_stats(individual_tolerance_matched_df, "XIV. Matched Individual Unmatched Entries with Tolerance in TDS and Tally", "TDS Deposited(Rs.)")

                    # Get remaining unmatched individual entries after removing tolerance matched ones
                    remaining_unmatched_tds_after_tolerance, remaining_unmatched_zoho_after_tolerance = get_remaining_unmatched_entries_with_tolerance(
                        remaining_unmatched_tds,
                        remaining_unmatched_zoho,
                        individual_tolerance_matched_df,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year'
                    )

                    # Display remaining unmatched individual entries after tolerance matching
                    display_dataframe_with_stats(remaining_unmatched_tds_after_tolerance, "XV. Remaining Unmatched Individual Entries After Tolerance in TDS", "TDS Deposited(Rs.)")
                    display_dataframe_with_stats(remaining_unmatched_zoho_after_tolerance, "XVI. Remaining Unmatched Individual Entries After Tolerance in Tally", "tds of the current fin. year")

                    # Perform matching based on the first three words in the deductor name
                    three_words_matched_df = match_individual_entries_based_on_three_words(
                        remaining_unmatched_tds_after_tolerance,
                        remaining_unmatched_zoho_after_tolerance,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year'
                    )
                    display_dataframe_with_stats(three_words_matched_df, "XVII. Matched Individual Unmatched Entries Based on Three Words in Deductor Name", "TDS Deposited(Rs.)")

                    # Get remaining unmatched individual entries after removing matches based on three words in the deductor name
                    remaining_unmatched_tds_after_three_words, remaining_unmatched_zoho_after_three_words = get_remaining_unmatched_entries_after_three_words(
                        remaining_unmatched_tds_after_tolerance,
                        remaining_unmatched_zoho_after_tolerance,
                        three_words_matched_df,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year'
                    )

                    # Display remaining unmatched individual entries after matching based on three words
                    display_dataframe_with_stats(remaining_unmatched_tds_after_three_words, "XVIII. Remaining Unmatched Individual Entries After Three Words Matching in TDS", "TDS Deposited(Rs.)")
                    display_dataframe_with_stats(remaining_unmatched_zoho_after_three_words, "XIX. Remaining Unmatched Individual Entries After Three Words Matching in Tally", "tds of the current fin. year")

                    # Perform tolerance matching on individual unmatched entries based on three words in the deductor name
                    three_words_tolerance_matched_df = match_individual_entries_with_tolerance_based_on_three_words(
                        remaining_unmatched_tds_after_three_words,
                        remaining_unmatched_zoho_after_three_words,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year',
                        tolerance=10
                    )

                    # Display DataFrame and set session state
                    display_dataframe_with_stats(three_words_tolerance_matched_df, "XX. Matched Individual Unmatched Entries with Tolerance Based on Three Words in Deductor Name", "TDS Deposited(Rs.)")

                    # Get remaining unmatched individual entries after removing tolerance matched ones based on three words
                    remaining_unmatched_tds_after_tolerance_three_words, remaining_unmatched_zoho_after_tolerance_three_words = get_remaining_unmatched_entries_after_tolerance_three_words(
                        remaining_unmatched_tds_after_three_words,
                        remaining_unmatched_zoho_after_three_words,
                        three_words_tolerance_matched_df,
                        'Name of Deductor',
                        'name of the deductor',
                        'TDS Deposited(Rs.)',
                        'tds of the current fin. year'
                    )

                    # Display remaining unmatched individual entries after tolerance matching based on three words
                    display_dataframe_with_stats(remaining_unmatched_tds_after_tolerance_three_words, "XXI. Remaining Unmatched Individual Entries After Tolerance Based on Three Words in TDS", "TDS Deposited(Rs.)")
                    display_dataframe_with_stats(remaining_unmatched_zoho_after_tolerance_three_words, "XXII. Remaining Unmatched Individual Entries After Tolerance Based on Three Words in Tally", "tds of the current fin. year")

                    # Create Final DataFrame with All Matched Entries
                    final_matched_df = pd.concat([
                        exact_matching_df,
                        tolerance_matching_df,
                        individual_matched_df,
                        individual_tolerance_matched_df,
                        three_words_matched_df,
                        three_words_tolerance_matched_df
                    ]).reset_index(drop=True)

                    # Save final matched DataFrame to session state
                    st.session_state.final_matched_df = final_matched_df

                    # Save final unmatched DataFrames to session state
                    st.session_state.remaining_unmatched_tds = remaining_unmatched_tds_after_tolerance_three_words
                    st.session_state.remaining_unmatched_zoho = remaining_unmatched_zoho_after_tolerance_three_words

                    # Selecting only the required columns (index 0 and 4)
                    # final_matched_df_selected = st.session_state.final_matched_df.iloc[:, [0,2,4]]
                    final_matched_df_selected = st.session_state.final_matched_df.iloc[:, [0, 4]]

                    # Displaying the updated DataFrame with only the selected columns
                    display_dataframe_with_stats(final_matched_df_selected, "XXIII. Final Consolidated Matched Entries (Selected Columns)", "TDS Deposited(Rs.)")

                    # display_dataframe_with_stats(final_matched_df, "Consolidated matched data.", "TDS Deposited(Rs.)")

                    # Initialize session state for 'final_matched_df_selected'
                    st.session_state.final_matched_df_selected = final_matched_df_selected

                    # Ensure summary tables are created only when all required data is available
                    if st.session_state.final_matched_df is not None and not st.session_state.remaining_unmatched_tds.empty and not st.session_state.remaining_unmatched_zoho.empty:
                        count_summary_df, amount_summary_df = create_summary_tables(
                            st.session_state.final_matched_df,
                            st.session_state.remaining_unmatched_tds,
                            st.session_state.remaining_unmatched_zoho,
                            st.session_state.selected_columns,
                            st.session_state.df
                        )

                        # Ensure these variables are checked before calling display_summary_tables
                        if count_summary_df is not None and amount_summary_df is not None:
                            display_summary_tables(count_summary_df, amount_summary_df)


                    # Perform matching and reconciliation logic here...
                    # [Insert your matching logic from here onwards]

            except Exception as e:
                st.error(f"An error occurred while processing files: {str(e)}")
        else:
            st.sidebar.write("Awaiting file upload...")

    # Place download buttons after file processing to prevent reloads
    if st.session_state.df is not None:
        st.sidebar.download_button("(I). Download Individual 26AS", convert_df_to_excel(st.session_state.df), "Individual_26AS.xlsx")
    if st.session_state.aggregated_tds_df is not None:
        st.sidebar.download_button("(II). Download Aggregated 26AS", convert_df_to_excel(st.session_state.aggregated_tds_df), "Aggregated_26AS.xlsx")
    if st.session_state.selected_columns is not None:
        st.sidebar.download_button("(III). Download Individual Tally", convert_df_to_excel(st.session_state.selected_columns), "Individual_Zoho.xlsx")
    if st.session_state.aggregated_zoho_df is not None:
        st.sidebar.download_button("(IV). Download Aggregated Tally", convert_df_to_excel(st.session_state.aggregated_zoho_df), "Aggregated_Zoho.xlsx")
    if not st.session_state.remaining_unmatched_tds.empty:
        st.sidebar.download_button("(XXI). Download Final Unmatched TDS", convert_df_to_excel(st.session_state.remaining_unmatched_tds), "Unmatched_TDS.xlsx")
    if not st.session_state.remaining_unmatched_zoho.empty:
        st.sidebar.download_button("(XXII). Download Final Unmatched Tally", convert_df_to_excel(st.session_state.remaining_unmatched_zoho), "Unmatched_Zoho.xlsx")
    if st.session_state.final_matched_df_selected is not None:
        st.sidebar.download_button("(XXIII). Download Final Matching Data", convert_df_to_excel(st.session_state.final_matched_df_selected), "Exact_Matches.xlsx")

if __name__ == "__main__":
    main()


    