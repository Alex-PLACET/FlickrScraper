# FlickrScraper

This is a simple Python script that scrapes images from Flickr. It uses the Flickr API to search for images/metadata based on a search query and download them to a specified directory.

## Installation

1. Clone the repository
2. Install the required packages:
Using uv
```bash
uv sync
```

## Usage

1. Create a Flickr API key and secret by following the instructions [here](https://www.flickr.com/services/api/keys/)

2. You can run the script using the following command:
```bash
python flickr_scraper.py --api_key <API_KEY> --api_secret <API_SECRET> --tags <TAGS> --output_dir <OUTPUT_DIR>
```

Or you can use a JSON configuration file:
```json
{
    "api_key": "<API_KEY>",
    "api_secret": "<API_SECRET>",
    "output_dir": "<OUTPUT_DIR>"
}
```

To use the detection of the sun or the moon, you have to create an account on (roboflow)[https://app.roboflow.com/] and use the API key with the option `--roboflow_api_key <ROBOFLOW_API_KEY>`
