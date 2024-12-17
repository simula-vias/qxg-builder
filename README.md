# QXG-Builder
QXG-Builder is an approach for building spatio-temporal graphs from scenes in automated driving.
## Installation
Install required libraries in requirement.txt file.
```pip install -r requirements.txt```
## NuScnes Dataset
Download the nuscenes dataset in the working directory by executing this script (Mentioned in : https://www.nuscenes.org/nuscenes?tutorial=nuscenes):
```
# !mkdir -p /data/sets/nuscenes  # Make the directory to store the nuScenes dataset in.

# !wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.

# !tar -xf v1.0-mini.tgz -C /data/sets/nuscenes  # Uncompress the nuScenes mini split.
```
## Execution Arguments:
``` 
-s : sensor ["LIDAR_TOP","CAM_FRONT","CAM_FRONT_RIGHT","CAM_FRONT_LEFT","CAM_BACK","CAM_BACK_RIGHT","CAM_BACK_LEFT"]

-mode : indivudal or temporal (Individual is for seprate frame by frame graphs, Temporal is the graph for the whole scene)

-d: dataroot of the downloaded dataset 

-v: version of the dataset (v1.0-mini)
``` 

### Individual mode
This mode gives independent graphs for each frame.
```python Main.py individual -d /data/sets/nuscenes -v v1.0-mini```
### Temporal mode
This mode gives independent graphs for each frame.
```python Main.py temporal -d /data/sets/nuscenes -v v1.0-mini```

## Results:
For each selected sensor there will be a folder created in the directory ```results``` which will contain both the execution time and the graph.

## Docker

To build and run the container:
### Build the image
docker build -t qxg-builder .

### Run the container
docker run -it qxg-builder
Note: Since this project works with the NuScenes dataset, you'll need to mount the dataset directory when running the container in practice. You can do this by adding a volume mount:
```
docker run -it -v /path/to/your/nuscenes/dataset:/data/sets/nuscenes qxg-builder
```
The path /data/sets/nuscenes matches the default dataroot path shown in `Main.py`:
```
        "-d",
        "--dataroot",
        default="./data/sets/nuscenes",
        help="Path of nuScenes dataset",
```