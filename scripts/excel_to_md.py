#!/usr/bin/env python3
"""
Convert Excel (.xlsx) to Markdown (.md) or Text (.txt).
Usage:
    python scripts/excel_to_md.py file.xlsx [output.md|output.txt]
    python scripts/excel_to_md.py Research_Project_Plan_15hrs_week.xlsx Research_Plan.md
"""

import sys
import os
from pathlib import Path

def excel_to_markdown(filepath, output_path=None):
    """Convert Excel to Markdown table format."""
    try:
        import pandas as pd
        from tabulate import tabulate
        
        # Read Excel (handle multiple sheets)
        excel_file = pd.ExcelFile(filepath)
        sheet_names = excel_file.sheet_names
        
        # Determine output path
        if output_path is None:
            base = Path(filepath).stem
            output_path = f"{base}.md"
        
        md_content = f"# {Path(filepath).stem}\n\n"
        
        # Process each sheet
        for sheet_name in sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            if len(sheet_names) > 1:
                md_content += f"## Sheet: {sheet_name}\n\n"
            
            # Convert to markdown using tabulate for better formatting
            md_content += tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            md_content += "\n\n"
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        total_rows = sum(pd.read_excel(filepath, sheet_name=s).shape[0] for s in sheet_names)
        print(f"[OK] Converted to: {output_path}")
        print(f"  Sheets: {len(sheet_names)}, Total Rows: {total_rows}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def excel_to_text(filepath, output_path=None):
    """Convert Excel to plain text table."""
    try:
        import pandas as pd
        from tabulate import tabulate
        
        # Read Excel (handle multiple sheets)
        excel_file = pd.ExcelFile(filepath)
        sheet_names = excel_file.sheet_names
        
        # Determine output path
        if output_path is None:
            base = Path(filepath).stem
            output_path = f"{base}.txt"
        
        text_content = f"{Path(filepath).stem}\n"
        text_content += "=" * 80 + "\n\n"
        
        # Process each sheet
        for sheet_name in sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            if len(sheet_names) > 1:
                text_content += f"Sheet: {sheet_name}\n"
                text_content += "-" * 80 + "\n\n"
            
            # Convert to text using tabulate for better formatting
            text_content += tabulate(df, headers='keys', tablefmt='grid', showindex=False)
            text_content += "\n\n"
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        total_rows = sum(pd.read_excel(filepath, sheet_name=s).shape[0] for s in sheet_names)
        print(f"[OK] Converted to: {output_path}")
        print(f"  Sheets: {len(sheet_names)}, Total Rows: {total_rows}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/excel_to_md.py <file.xlsx> [output.md|output.txt]")
        print("\nExamples:")
        print("  python scripts/excel_to_md.py file.xlsx file.md")
        print("  python scripts/excel_to_md.py file.xlsx file.txt")
        print("  python scripts/excel_to_md.py file.xlsx  # auto: file.md")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if output_path and output_path.endswith(".txt"):
        excel_to_text(filepath, output_path)
    else:
        excel_to_markdown(filepath, output_path)
