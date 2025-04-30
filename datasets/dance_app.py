import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# Load the dataset
# Note: other available arguments include 'split', 'max_samples', etc
dataset = fouh.load_from_hub("dgural/DanceTrack")

# Launch the App
session = fo.launch_app(dataset)