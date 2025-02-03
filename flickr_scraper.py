import json
import time
from datetime import datetime, date
from typing import Final, List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
import flickrapi
from loguru import logger
from cyclopts import App, Parameter
from typing import Annotated
import requests

from inference_sdk import InferenceHTTPClient


@dataclass
class FlickrConfig:
    api_key: str
    api_secret: str
    output_folder: Path
    camera_model: Optional[str] = None
    tags: Optional[List[str]] = None
    save_images: bool = True
    queries_per_hour: int = 3600
    min_date: Optional[date] = None
    detect_sun: bool = False
    detect_moon: bool = False
    roboflow_api_url: str = "https://detect.roboflow.com"
    roboflow_api_key: Optional[str] = None


class FlickrDownloader:
    def __init__(self, config: FlickrConfig):
        self.config = config
        self.flickr = flickrapi.FlickrAPI(
            config.api_key, config.api_secret, format="parsed-json", cache=True
        )
        self.output_folder = Path(config.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.metadata_folder: Path = self.output_folder / "metadata"
        self.metadata_folder.mkdir(exist_ok=True)
        self.images_folder: Path = self.output_folder / "images"
        if self.config.save_images:
            self.images_folder.mkdir(exist_ok=True)
        self.roboflow_client_inference = InferenceHTTPClient(
            api_url=config.roboflow_api_url, api_key=config.roboflow_api_key
        )

        # Setup loguru
        log_file: Final[Path] = self.output_folder / "downloader.log"
        logger.add(
            log_file,
            rotation="10 MB",
            retention="1 week",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )

    @sleep_and_retry
    @limits(calls=9999999999, period=3600)
    def _get_photo_info(self, photo_id: str) -> Dict:
        """Get detailed information about a specific photo."""
        info = self.flickr.photos.getInfo(photo_id=photo_id)
        exif = self.flickr.photos.getExif(photo_id=photo_id)
        info["photo"]["exif"] = exif.get("photo", {}).get("exif", [])
        return info

    @sleep_and_retry
    @limits(calls=10, period=10)
    def _detect(self, image_url: str, model_id: str) -> bool:
        """Detect object in image using Roboflow model."""
        result = self.roboflow_client_inference.infer(image_url, model_id=model_id)
        if type(result) is not dict:
            raise ValueError("Unexpected response from Roboflow API")
        return len(result["predictions"]) > 0

    def _detect_sun(self, image_url: str) -> bool:
        """Detect sun in image using Roboflow model."""
        return self._detect(image_url, "sun-detection-bef5i/1")

    def _detect_moon(self, image_url: str) -> bool:
        """Detect moon in image using Roboflow model."""
        return self._detect(image_url, "moon-detection-3k8i/1")

    def _download_image(self, photo_id: str, url: str) -> Path:
        """Download an image from Flickr using flickrapi."""

        image_path = self.images_folder / f"{photo_id}.jpg"
        # download the image
        response = requests.get(url)
        response.raise_for_status()
        with open(image_path, "wb") as f:
            f.write(response.content)

        logger.debug(f"Successfully downloaded image {photo_id}")
        return image_path

    def _save_metadata(self, photo_id: str, metadata: Dict) -> None:
        """Save photo metadata to JSON file."""
        try:
            metadata_path = self.metadata_folder / f"{photo_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata for photo {photo_id}")
        except Exception as e:
            logger.error(f"Error saving metadata for photo {photo_id}: {e}")

    def _process_photo(self, photo: Dict) -> None:
        """Process a single photo: get info, download image, and save metadata."""
        photo_id = photo["id"]
        logger.info(f"Processing photo {photo_id}")

        try:
            info = self._get_photo_info(photo_id)
            if not info:
                return

            if self.config.min_date:
                upload_date = datetime.fromtimestamp(
                    int(info["photo"]["dateuploaded"])
                ).date()
                if upload_date < self.config.min_date:
                    logger.debug(
                        f"Skipping photo {photo_id} - uploaded before {self.config.min_date}"
                    )
                    return

            # Check camera model
            exif_data = info["photo"].get("exif", [])
            maker: Optional[str] = None
            model: Optional[str] = None

            for item in exif_data:
                if item.get("tag") == "Make":
                    maker = item.get("raw", {}).get("_content")
                elif item.get("tag") == "Model":
                    model = item.get("raw", {}).get("_content")

            if maker or model:
                logger.info(f"Camera maker: {maker}, model: {model}")

            if self.config.camera_model and model != self.config.camera_model:
                logger.debug(
                    f"Skipping photo {photo_id} - camera model {model} does not match {self.config.camera_model}"
                )
                return

            if model:
                self._save_metadata(photo_id, info)
                if self.config.save_images:
                    sizes = self.flickr.photos.getSizes(photo_id=photo_id)
                    # Check that "Original" size is available
                    original_size = next(
                        (
                            size
                            for size in sizes["sizes"]["size"]
                            if size["label"] == "Original"
                        ),
                        None,
                    )
                    if not original_size:
                        logger.debug(
                            f"Skipping photo {photo_id} - no original size available"
                        )
                        return
                    # check "Large 2048" then "Large 1600" then "Large"
                    sizes["sizes"]["size"].reverse()
                    image_large_size: Optional[str] = next(
                        (
                            size
                            for size in sizes["sizes"]["size"]
                            if size["label"] in ["Large 2048", "Large 1600", "Large"]
                        ),
                        None,
                    )
                    if not image_large_size:
                        logger.debug(
                            f"Skipping photo {photo_id} - no large size available"
                        )
                        return
                    image_url = image_large_size["source"]
                    if self.config.detect_sun and not self._detect_sun(image_url):
                        logger.debug(f"Skipping photo {photo_id} - no sun detected")
                        return
                    if self.config.detect_moon and not self._detect_moon(image_url):
                        logger.debug(f"Skipping photo {photo_id} - no moon detected")
                        return
                    try:
                        original_source = (
                            original_size["source"] if original_size else None
                        )
                        if not original_source:
                            raise ValueError(
                                f"Original source not found for photo {photo_id}"
                            )
                        image_path: Path = self._download_image(
                            photo_id, original_source
                        )
                    except Exception as e:
                        logger.error(f"Error downloading image {photo_id}: {e}")
                        return
        except Exception as e:
            logger.error(f"Error processing photo {photo_id}: {e}")

    def _get_photos(self, page_number: int) -> Dict:
        """Search for photos based on configuration."""
        search_params = {
            "media": "photos",
            "per_page": 500,
            "extras": "url_o,media,date_upload",
            "content_types": 0,  # 0 for photos, 1 for screenshots, 2 for other
            "page": page_number,
        }

        if self.config.tags:
            search_params["tags"] = ",".join(self.config.tags)
            search_params["tag_mode"] = "all"

        if self.config.min_date:
            search_params["min_upload_date"] = self.config.min_date.strftime("%Y-%m-%d")

        return self.flickr.photos.search(**search_params)

    def _get_list_of_existing_images(self) -> List[str]:
        """Get a list of existing images in the output folder."""
        return [
            f.name.split(".")[0]
            for f in self.images_folder.glob("*.jpg")
            if f.is_file()
        ]

    def run(self) -> None:
        """Main execution method."""
        start_time = time.time()
        logger.info("Starting Flickr download process...")

        page_number: int = 1
        photos: Dict = self._get_photos(page_number)
        photos_elements = photos["photos"]["photo"]
        pages_count: Final[int] = photos["photos"]["pages"]
        list_of_existing_images: Final[List[str]] = self._get_list_of_existing_images()
        filtered_photos_elements = [
            photo
            for photo in photos_elements
            if photo["id"] not in list_of_existing_images
        ]

        for page_number in range(2, pages_count + 1):
            with ThreadPoolExecutor(max_workers=25) as executor:
                list(executor.map(self._process_photo, filtered_photos_elements))
            photos = self._get_photos(page_number)
            photos_elements = photos["photos"]["photo"]
            filtered_photos_elements = [
                photo
                for photo in photos_elements
                if photo["id"] not in list_of_existing_images
            ]

        elapsed_time = time.time() - start_time
        logger.info(f"Download process completed in {elapsed_time:.2f} seconds")


def load_config_file(config_path: Path) -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")


# Create cyclopts app
app = App(
    name="flickr-downloader",
    help="Download images and metadata from Flickr based on specified criteria",
)


@app.default
def main(
    api_key: Annotated[str, Parameter(help="Flickr API key")],
    api_secret: Annotated[str, Parameter(help="Flickr API secret")],
    output_folder: Annotated[Path, Parameter(help="Output folder")] = Path(
        "flickr_output"
    ),
    camera_model: Annotated[
        Optional[str], Parameter(help="Filter by camera model")
    ] = None,
    tags: Annotated[
        Optional[List[str]],
        Parameter(help="List of tags to filter photos"),
    ] = None,
    metadata_only: Annotated[
        bool, Parameter(help="Only download metadata, skip images")
    ] = False,
    queries_per_hour: Annotated[
        int, Parameter(help="Maximum number of API queries per hour")
    ] = 3600,
    min_date: Annotated[
        Optional[str], Parameter(help="Minimum upload date (YYYY-MM-DD)")
    ] = None,
    roboflow_api_url: Annotated[
        str, Parameter(help="Roboflow API URL")
    ] = "https://detect.roboflow.com",
    roboflow_api_key: Annotated[
        Optional[str], Parameter(help="Roboflow API key")
    ] = None,
    detect_sun: Annotated[
        bool,
        Parameter(
            help="Detect sun in images. If the sun is not detected, the image is not downloaded. Requires Roboflow API key"
        ),
    ] = False,
    detect_moon: Annotated[
        bool,
        Parameter(
            help="Detect moon in images. If the moon is not detected, the image is not downloaded. Requires Roboflow API key"
        ),
    ] = False,
    config_file: Annotated[
        Optional[Path], Parameter(help="Path to JSON configuration file")
    ] = None,
):
    """Download images and m etadata from Flickr."""
    # Load configuration
    if config_file:
        config_data = load_config_file(config_file)
        # Override config file with command line arguments
        if api_key:
            config_data["api-key"] = api_key
        if api_secret:
            config_data["api-secret"] = api_secret
        if output_folder:
            config_data["output-folder"] = output_folder
        if camera_model:
            config_data["camera-model"] = camera_model
        if tags:
            config_data["tags"] = tags
        if metadata_only:
            config_data["save-images"] = not metadata_only
        if queries_per_hour:
            config_data["queries-per-hour"] = queries_per_hour
        if detect_sun:
            config_data["detect-sun"] = detect_sun
        if detect_moon:
            config_data["detect-moon"] = detect_moon
        if min_date:
            config_data["min-date"] = min_date
        if roboflow_api_url:
            config_data["roboflow-api-url"] = roboflow_api_url
        if roboflow_api_key:
            config_data["roboflow-api-key"] = roboflow_api_key
    else:
        # Use command line arguments
        if not api_key or not api_secret:
            raise ValueError(
                "API key and secret are required if no config file is provided"
            )

        if detect_moon or detect_sun:
            if not roboflow_api_key:
                raise ValueError(
                    "Roboflow API key is required for sun and moon detection"
                )

        config_data = {
            "api-key": api_key,
            "api-secret": api_secret,
            "output-folder": output_folder,
            "camera-model": camera_model,
            "tags": tags,
            "save-images": not metadata_only,
            "queries-per-hour": queries_per_hour,
            "min-date": min_date,
            "roboflow-api-url": roboflow_api_url,
            "roboflow-api-key": roboflow_api_key,
            "detect-sun": detect_sun,
            "detect-moon": detect_moon,
        }

    # Parse date if provided
    min_date_obj = None
    if config_data.get("min-date"):
        min_date_obj = parse_date(config_data["min-date"])

    # Create configuration object
    config: Final[FlickrConfig] = FlickrConfig(
        api_key=config_data["api-key"],
        api_secret=config_data["api-secret"],
        output_folder=Path(config_data["output-folder"]),
        camera_model=config_data.get("camera-model"),
        tags=config_data.get("tags"),
        save_images=config_data.get("save-images", True),
        queries_per_hour=config_data.get("queries-per-hour", 3600),
        min_date=min_date_obj,
        roboflow_api_url=config_data.get(
            "roboflow-api-url", "https://detect.roboflow.com"
        ),
        roboflow_api_key=config_data.get("roboflow-api-key"),
        detect_sun=config_data.get("detect-sun", False),
        detect_moon=config_data.get("detect-moon", False),
    )

    # Create and run downloader
    downloader = FlickrDownloader(config)
    downloader.run()


if __name__ == "__main__":
    app()
