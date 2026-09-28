"""Prepare UAV-PDD2023 as a pothole-only YOLO dataset. See README.md."""
from pothole_converter import main
if __name__ == "__main__":
    main(default_name_regex=r"pothole|potholes|ph|pit", dataset_label="UAV-PDD2023")
