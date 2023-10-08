# type: ignore
# flake8: noqa
#
#
#
#
#
import os

# Directory containing .mmd files
directory = "/Users/robinpfaff/Library/CloudStorage/OneDrive-ZHAW/6_Semester/BA/03_vs_code/mermaid"

# Output directory for PDF files
output_directory = "../text/05_method_files"

# Bash script file name
script_file = "generate_diagrams.sh"

# List to store the generated strings
command_strings = []

# List all .mmd file names in the directory
mmd_files = [filename for filename in os.listdir(directory) if filename.endswith(".mmd")]

# Generate the command strings
for mmd_file in mmd_files:
    input_file = os.path.join(directory, mmd_file)
    output_file = os.path.join(output_directory, os.path.splitext(mmd_file)[0] + ".pdf")  # Use the output directory
    command_string = f"mmdc -i {input_file} -o {output_file} -e pdf -C my.css -f"
    command_strings.append(command_string)

# Create the Bash script
with open(script_file, "w") as script:
    script.write("#!/bin/bash\n")
    for command in command_strings:
        script.write(command + "\n")

# Make the Bash script executable
os.chmod(script_file, 0o755)

print(f"Bash script '{script_file}' generated and made executable.")


#
#
#
