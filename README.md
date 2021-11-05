# SubmapsRegistration

Tools for keypoint detection, feature computation and correspondance matching in point clouds, with the focus on bathymetric data.

![](img/gicp.gif)

## Dependencies (Ubuntu 20.04)
* PCL  http://pointclouds.org/
* EIGEN http://eigen.tuxfamily.org/
* Auvlib https://github.com/nilsbore/auvlib
* ymal-cpp https://github.com/jbeder/yaml-cpp

## Building

Clone this repository and create a `build` folder under the root, then execute
```
cd build
cmake ..
make -j4
```

## Running
Avalible under the `bin` folder:

Registration of a pointcloud against itself after a random transformation
```
./submap_registration ../meshes/submap_1.pcd ../config.yaml
```
And hit 'q' on the window to go through the registration steps:
1. Keypoints extraction
2. Correspondance matching between the keypoints features
3. Initial alingment based on the correspondances
4. GICP registration
5. Exit

Constructs submaps from a bathymetry survey with 'submap_size' pings per submap and saves them as 'pcd' files in the output folder
```
./submaps_construction --submap_size 200 --mbes_cereal /path/to/file.cereal --output_folder /path/to/output/folder
```

Util for visualizing the output submaps within the range [first_submap,last_submap] from the previous app one by one
```
./visualize_submaps --input_folder /path/to/folder/with/pcd --first_submap 200 --last_submap 250
```

## Tuning
There are some important parameters to tune, they are written in `config.yaml`. So that you do not need to build every time you change the parameters during tuning. 
