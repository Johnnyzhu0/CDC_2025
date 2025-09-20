import pandas as pd
from analyze_space_data import SpaceEconomyAnalyzer
import datetime

def export_data_to_txt(analyzer, output_file="data_export.txt"):
    """
    Export analysis results and data summaries to a text file
    
    Args:
        analyzer: SpaceEconomyAnalyzer instance
        output_file: Name of the output text file
    """
    
    with open(output_file, 'w') as f:
        # Header
        f.write("SPACE ECONOMY DATA EXPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: Business.xlsx\n\n")
        
        # Table summary
        f.write("TABLE SUMMARY\n")
        f.write("-" * 30 + "\n")
        for table_name, df in analyzer.tables.items():
            f.write(f"{table_name}:\n")
            f.write(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
            f.write(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")
            f.write(f"  Missing values: {df.isnull().sum().sum()}\n\n")
        
        # Sample data from each table
        f.write("\nSAMPLE DATA\n")
        f.write("=" * 50 + "\n")
        
        for table_name, df in analyzer.tables.items():
            f.write(f"\n{table_name.upper()}\n")
            f.write("-" * len(table_name) + "\n")
            
            # Write column names
            f.write("Columns:\n")
            for i, col in enumerate(df.columns[:5], 1):  # First 5 columns
                f.write(f"  {i}. {col}\n")
            if len(df.columns) > 5:
                f.write(f"  ... and {len(df.columns) - 5} more columns\n")
            
            # Write first few rows of data
            f.write("\nFirst 3 rows:\n")
            for idx, row in df.head(3).iterrows():
                f.write(f"Row {idx + 1}: {str(row.iloc[0])[:100]}...\n")
            f.write("\n")

def export_specific_table_to_txt(analyzer, table_name, output_file=None):
    """
    Export a specific table's data to a text file
    
    Args:
        analyzer: SpaceEconomyAnalyzer instance
        table_name: Name of the table to export
        output_file: Output file name (optional)
    """
    
    if output_file is None:
        output_file = f"{table_name.replace(' ', '_').lower()}_data.txt"
    
    if table_name not in analyzer.tables:
        print(f"Table '{table_name}' not found!")
        return
    
    df = analyzer.tables[table_name]
    
    with open(output_file, 'w') as f:
        f.write(f"DATA EXPORT: {table_name.upper()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
        
        # Column information
        f.write("COLUMNS:\n")
        f.write("-" * 20 + "\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:2d}. {col}\n")
        
        f.write(f"\nDATA:\n")
        f.write("-" * 20 + "\n")
        
        # Write all data
        for idx, row in df.iterrows():
            f.write(f"\nRow {idx + 1}:\n")
            for col_idx, value in enumerate(row):
                if pd.notna(value):  # Only write non-null values
                    f.write(f"  {df.columns[col_idx]}: {value}\n")

def append_analysis_to_txt(text_content, output_file="data_analysis_results.txt"):
    """
    Append analysis results to the existing text file
    
    Args:
        text_content: String content to append
        output_file: Text file to append to
    """
    
    with open(output_file, 'a') as f:
        f.write(f"\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Analysis Update\n")
        f.write("-" * 50 + "\n")
        f.write(text_content)
        f.write("\n" + "=" * 50 + "\n")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SpaceEconomyAnalyzer()
    analyzer.import_all_tables()
    
    # Export summary to text
    export_data_to_txt(analyzer, "space_economy_summary.txt")
    print("✅ Exported summary to space_economy_summary.txt")
    
    # Export specific table
    export_specific_table_to_txt(analyzer, "Table 1", "value_added_data.txt")
    print("✅ Exported Table 1 to value_added_data.txt")
    
    # Append custom analysis
    analysis_text = """
    PRELIMINARY FINDINGS:
    
    1. The space economy data spans from 2012 to 2023
    2. Multiple economic indicators are tracked:
       - Real Value Added
       - Gross Output
       - Employment levels
       - Compensation data
    
    3. Data is broken down by industry sectors
    4. Price indexes show inflation-adjusted trends
    
    NEXT STEPS:
    - Analyze year-over-year growth trends
    - Identify top-performing industry sectors
    - Calculate employment productivity metrics
    """
    
    append_analysis_to_txt(analysis_text)
    print("✅ Added analysis to data_analysis_results.txt")