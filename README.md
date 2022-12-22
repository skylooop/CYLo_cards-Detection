## cYLo_Cards-detection
Most of the code is taken from official ![Yolo-v5](https://github.com/ultralytics/yolov5) repository


The idea is to implement self-supervised learning for cards, on which object detection model (like YoLo) was not trained. If threshold given by Yolo is small for card (thus most likely it was not in training dataset), then label is given by the most "similiar" card.

In roboflow folder you can find script for downloading different datasets into datasets folder like this:

```
python roboflow/get_dataset.py --dataset_name "name of dataset" --api_key "your api key from roboflow" \\
                               --workspace_name "name of workspace" --project "name of project" \\
                                --model "yolov5"
```