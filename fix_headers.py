import pandas as pd
import numpy as np

def import_table_with_proper_headers(file_path="Business.xlsx", sheet_name="Table 1"):
    """
    Import a table with proper column headers by skipping metadata rows
    
    Args:
        file_path: Path to Excel file
        sheet_name: Name of the sheet to import
    
    Returns:
        pd.DataFrame: Properly formatted DataFrame
    """
    
    # Read the raw data first to understand structure
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    # Find the row with years (this will be our header row)
    header_row = None
    for i, row in raw_df.iterrows():
        # Look for a row that contains years like 2012, 2013, etc.
        row_str = str(row.values)
        if '2012' in row_str and '2023' in row_str:
            header_row = i
            break
    
    if header_row is None:
        print("Could not find header row with years. Using default import.")
        return pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Re-read with proper header row
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
    
    # Clean up the DataFrame
    # Remove rows before the actual data
    data_start_row = header_row + 1
    df = df.iloc[1:].reset_index(drop=True)  # Skip the header row itself
    
    # Clean column names
    new_columns = []
    for col in df.columns:
        if pd.isna(col):
            new_columns.append('Index')
        elif str(col).startswith('Unnamed'):
            new_columns.append('Industry')
        else:
            new_columns.append(str(col))
    
    df.columns = new_columns
    
    # Clean the data
    df = df.dropna(how='all')  # Remove completely empty rows
    
    return df

def export_clean_table_to_txt(sheet_name="Table 1", output_file=None):
    """
    Export a properly formatted table to text file
    """
    if output_file is None:
        output_file = f"{sheet_name.replace(' ', '_').lower()}_clean.txt"
    
    # Import with proper headers
    df = import_table_with_proper_headers(sheet_name=sheet_name)
    
    with open(output_file, 'w') as f:
        f.write(f"CLEAN DATA EXPORT: {sheet_name.upper()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
        
        # Column information
        f.write("COLUMNS:\n")
        f.write("-" * 20 + "\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:2d}. {col}\n")
        
        f.write(f"\nSAMPLE DATA (first 10 rows):\n")
        f.write("-" * 40 + "\n")
        
        # Write sample data in a readable format
        for idx, row in df.head(10).iterrows():
            f.write(f"\nRow {idx + 1}:\n")
            for col_name, value in row.items():
                if pd.notna(value) and str(value).strip() != '':
                    f.write(f"  {col_name}: {value}\n")
        
        f.write(f"\n\nFULL DATA:\n")
        f.write("-" * 20 + "\n")
        # Export as CSV-like format for readability
        f.write(df.to_string(index=False))
    
    print(f"✅ Exported clean {sheet_name} data to {output_file}")
    return df

if __name__ == "__main__":
    # Generate clean exports for all tables
    print("Generating clean exports for all tables...")
    print("=" * 50)
    
    tables_to_process = [
        ("Table 1", "Real Value Added"),
        ("Table 2", "Value Added"), 
        ("Table 3", "Value Added Price Indexes"),
        ("Table 4", "Real Gross Output"),
        ("Table 5", "Gross Output"),
        ("Table 6", "Gross Output Price Indexes"),
        ("Table 7", "Employment"),
        ("Table 8", "Compensation")
    ]
    
    for table_name, description in tables_to_process:
        try:
            print(f"\nProcessing {table_name} ({description})...")
            df = import_table_with_proper_headers(sheet_name=table_name)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
            
            # Export clean version
            output_file = f"{table_name.replace(' ', '_').lower()}_clean.txt"
            export_clean_table_to_txt(table_name, output_file)
            
        except Exception as e:
            print(f"  ❌ Error processing {table_name}: {e}")
    
    print(f"\n✅ Finished processing all tables!")
    print("Clean files generated with proper headers and year columns.")