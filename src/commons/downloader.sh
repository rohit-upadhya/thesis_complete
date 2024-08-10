#!/bin/bash

# URL of the PDF file
url="https://ks.echr.coe.int/documents/d/echr-ks/guide_art_4_ara"

# Path to save the downloaded PDF
output_path="Guide_Art_2_ARA_Downloaded.pdf"

# Use wget to download the PDF file
wget -O "$output_path" "$url"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "PDF downloaded successfully and saved as $output_path"
else
    echo "Failed to download PDF"
fi
