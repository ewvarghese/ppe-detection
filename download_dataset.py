from roboflow import Roboflow

rf = Roboflow(api_key="seJOcwaaRqXMypb5pPSK")
project = rf.workspace("vargheses-workspace").project("ppe-detection-qlq3d-hcv0h")
version = project.version(1)
dataset = version.download("yolov8")

print("Dataset downloaded at:", dataset.location)