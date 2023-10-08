# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| label: tbl-mytable5
#| tbl-cap: mycaption
import pandas as pd

# Specify the Excel file path
excel_file = "../../06_data/overview.xlsx"

# Specify the sheet name
sheet_name = "Tabelle1"

columns_to_read = ["Title", "Year covered", "resolution", "data type"]

# Specify the starting row and ending row of the range you want to read (1-based index)
start_row = 1  # Adjust this to the first row you want to read
end_row = 10   # Adjust this to the last row you want to read

# Read the specified range of cells from the Excel file
df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=range(1, start_row), nrows=(end_row - start_row + 1), usecols=columns_to_read)

# Now, df contains the data from the specified range of cells in the Excel file


from IPython.display import HTML

# Convert the DataFrame to an HTML table
html_table = df.to_html(classes="table table-striped table-bordered", escape=False)

# Display the HTML table
HTML(html_table)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
