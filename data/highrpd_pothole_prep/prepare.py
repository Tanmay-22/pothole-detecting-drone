"""Prepare HighRPD as a pothole-only YOLO dataset. See README.md."""
from pothole_converter import main
if __name__ == "__main__":
    main(default_name_regex=r"pothole|potholes|pit|pits", dataset_label="HighRPD")
