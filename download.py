import requests
import warnings

# Suppress SSL warning
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# URL of the file to download
url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json"

# Path to save the downloaded file
output_path = "modules.json"

try:
    # Send GET request to the URL with SSL verification disabled
    response = requests.get(url, verify=False)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

    # Write content to a file
    with open(output_path, "wb") as file:
        file.write(response.content)
    
    print(f"File downloaded successfully and saved to {output_path}")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")

