## cYLo_Cards-detection

Trying to implement self-supervised learning for unobserved in training data cards. If threshold given by Yolo is small for card, then label is given by the most "similiar" card.
In roboflow folder you can find script for downloading different datasets into datasets folder like this:
```
python roboflow/get_dataset.py --dataset_name "name of dataset" --api_key "your api key from roboflow" \\
                               --workspace_name "name of workspace" --project "name of project" \\
                                --model "yolov5"
```