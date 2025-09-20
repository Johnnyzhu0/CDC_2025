import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional

class SpaceEconomyAnalyzer:
    """
    A class to handle importing and analyzing the U.S. Space Economy data
    """
    
    def __init__(self, file_path: str = "Business.xlsx"):
        """
        Initialize the analyzer with the Excel file
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = file_path
        self.tables = {}
        self.sheet_names = []
        
    def import_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Import all data tables from the Excel file
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with sheet names as keys and DataFrames as values
        """
        try:
            print(f"Importing all tables from {self.file_path}...")
            print("=" * 50)
            
            # Get all sheet names
            excel_file = pd.ExcelFile(self.file_path)
            self.sheet_names = excel_file.sheet_names
            
            # Import all sheets except ReadMe
            data_sheets = [sheet for sheet in self.sheet_names if sheet != 'ReadMe']
            
            for sheet_name in data_sheets:
                print(f"Importing {sheet_name}...")
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                self.tables[sheet_name] = df
                
                # Display basic info
                print(f"  - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  - Columns: {list(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}")
                print()
            
            print(f"Successfully imported {len(self.tables)} tables!")
            return self.tables
            
        except Exception as e:
            print(f"Error importing tables: {e}")
            return {}
    
    def get_table_summary(self) -> None:
        """
        Display a summary of all imported tables
        """
        if not self.tables:
            print("No tables imported yet. Run import_all_tables() first.")
            return
        
        print("TABLE SUMMARY")
        print("=" * 50)
        
        for table_name, df in self.tables.items():
            print(f"\n📊 {table_name}")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Show column names (first few)
            cols_to_show = df.columns[:4].tolist()
            if len(df.columns) > 4:
                cols_to_show.append(f"... +{len(df.columns)-4} more")
            print(f"   Columns: {cols_to_show}")
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                print(f"   ⚠️  Missing values: {missing_count}")
    
    def preview_table(self, table_name: str, rows: int = 5) -> None:
        """
        Preview a specific table
        
        Args:
            table_name (str): Name of the table to preview
            rows (int): Number of rows to display
        """
        if table_name not in self.tables:
            print(f"Table '{table_name}' not found. Available tables: {list(self.tables.keys())}")
            return
        
        df = self.tables[table_name]
        print(f"\n📋 PREVIEW: {table_name}")
        print("=" * 50)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nFirst {rows} rows:")
        print(df.head(rows))
        
        print(f"\nData types:")
        print(df.dtypes)
        
        # Show numeric summary if there are numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric summary:")
            print(df[numeric_cols].describe())
    
    def search_in_tables(self, search_term: str) -> Dict[str, List[str]]:
        """
        Search for a term across all table columns and data
        
        Args:
            search_term (str): Term to search for
            
        Returns:
            Dict[str, List[str]]: Dictionary with table names and matching columns
        """
        results = {}
        search_lower = search_term.lower()
        
        for table_name, df in self.tables.items():
            matches = []
            
            # Search in column names
            for col in df.columns:
                if search_lower in str(col).lower():
                    matches.append(f"Column: {col}")
            
            # Search in data (first few rows for performance)
            for col in df.columns:
                if df[col].dtype == 'object':  # Only search text columns
                    sample_data = df[col].head(10).astype(str).str.lower()
                    if sample_data.str.contains(search_lower, na=False).any():
                        matches.append(f"Data in: {col}")
            
            if matches:
                results[table_name] = matches
        
        return results
    
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Get a specific table
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            pd.DataFrame: The requested table or None if not found
        """
        return self.tables.get(table_name)
    
    def list_tables(self) -> List[str]:
        """
        List all available table names
        
        Returns:
            List[str]: List of table names
        """
        return list(self.tables.keys())
    
    def export_table(self, table_name: str, output_file: str, format: str = 'csv') -> None:
        """
        Export a specific table to a file
        
        Args:
            table_name (str): Name of the table to export
            output_file (str): Output file path
            format (str): Export format ('csv', 'excel', 'json')
        """
        if table_name not in self.tables:
            print(f"Table '{table_name}' not found.")
            return
        
        df = self.tables[table_name]
        
        try:
            if format.lower() == 'csv':
                df.to_csv(output_file, index=False)
            elif format.lower() == 'excel':
                df.to_excel(output_file, index=False)
            elif format.lower() == 'json':
                df.to_json(output_file, orient='records', indent=2)
            else:
                print(f"Unsupported format: {format}")
                return
            
            print(f"✅ Exported {table_name} to {output_file}")
            
        except Exception as e:
            print(f"Error exporting table: {e}")

def main():
    """
    Main function to demonstrate the SpaceEconomyAnalyzer
    """
    # Initialize the analyzer
    analyzer = SpaceEconomyAnalyzer("Business.xlsx")
    
    # Import all tables
    tables = analyzer.import_all_tables()
    
    if tables:
        # Show summary of all tables
        analyzer.get_table_summary()
        
        # Preview the first table
        if analyzer.list_tables():
            first_table = analyzer.list_tables()[0]
            analyzer.preview_table(first_table, rows=3)
        
        print("\n" + "=" * 50)
        print("🚀 USAGE EXAMPLES:")
        print("=" * 50)
        print("# Get a specific table:")
        print("df = analyzer.get_table('Table 1')")
        print()
        print("# Search for terms across all tables:")
        print("results = analyzer.search_in_tables('revenue')")
        print()
        print("# Preview any table:")
        print("analyzer.preview_table('Table 2')")
        print()
        print("# Export a table:")
        print("analyzer.export_table('Table 1', 'output.csv')")

if __name__ == "__main__":
    main()