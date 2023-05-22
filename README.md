# qxg-builder
QXG-Builder is an approach for building saptio-temporal graphs from scenes in automated driving.
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
