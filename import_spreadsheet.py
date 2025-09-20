import pandas as pd
import os

def import_xlsx_file(file_path, sheet_name=None):
    """
    Import an Excel (.xlsx) file into a pandas DataFrame
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str, optional): Name of the sheet to read. If None, reads the first sheet
    
    Returns:
        pandas.DataFrame: The imported data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the Excel file
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"Successfully imported sheet '{sheet_name}' from {file_path}")
        else:
            df = pd.read_excel(file_path)
            print(f"Successfully imported first sheet from {file_path}")
        
        # Display basic information about the imported data
        print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Column names: {list(df.columns)}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Show data types
        print("\nData types:")
        print(df.dtypes)
        
        return df
    
    except Exception as e:
        print(f"Error importing file: {e}")
        return None

def list_sheets_in_excel(file_path):
    """
    List all sheet names in an Excel file
    
    Args:
        file_path (str): Path to the Excel file
    
    Returns:
        list: List of sheet names
    """
    try:
        # Read all sheets to get sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"Available sheets in {file_path}:")
        for i, name in enumerate(sheet_names, 1):
            print(f"  {i}. {name}")
        return sheet_names
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Using the Business.xlsx file found in the directory
    file_path = "Business.xlsx"
    
    print("Excel File Import Example")
    print("=" * 40)
    
    # First, list all available sheets
    sheets = list_sheets_in_excel(file_path)
    
    if sheets:
        # Import the first sheet
        df = import_xlsx_file(file_path)
        
        # If you want to import a specific sheet, uncomment the line below
        # df = import_xlsx_file(file_path, sheet_name="Sheet1")
        
        if df is not None:
            # Example of basic data exploration
            print("\n" + "=" * 40)
            print("Basic Data Exploration")
            print("=" * 40)
            
            # Check for missing values
            print("\nMissing values per column:")
            print(df.isnull().sum())
            
            # Basic statistics for numeric columns
            print("\nBasic statistics:")
            print(df.describe())