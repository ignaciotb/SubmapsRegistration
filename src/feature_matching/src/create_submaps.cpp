#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>

#include "data_tools/std_data.h"

#include <feature_matching/submaps.hpp>
#include "feature_matching/cxxopts.hpp"
#include "feature_matching/utils_visualization.hpp"


using namespace Eigen;
using namespace std;

int main(int argc, char** argv){

    // Inputs
    std::string folder_str, input_path, output_path;
    int submap_size;
    double overlap_area;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
        ("help", "Print help")
        ("submap_size", "Number of pings per submap", cxxopts::value(submap_size))
        ("overlap", "Minimum overlapping area to consider overlap. 0-1", cxxopts::value(overlap_area))
        ("output_folder", "Output path to folder", cxxopts::value(output_path))
        ("mbes_cereal", "Input path to MBES pings in cereal file", cxxopts::value(input_path));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({ "", "Group" }) << endl;
        exit(0);
    }

    // Parse submaps from cereal file
    boost::filesystem::path submaps_path(input_path);
    std::cout << "Input data " << submaps_path << std::endl;

    boost::filesystem::path folder_path(output_path);
    std::cout << "Output folder " << folder_path << std::endl;

    std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(submaps_path);
    std::cout << "Number of pings in survey " << std_pings.size() << std::endl;

    // Visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    SubmapsVec traj_pings = parsePingsAUVlib(std_pings);
    SubmapsVec submaps_gt = createSubmaps(traj_pings, submap_size);

    // Save submaps to disk
    int i = 0;
    PointCloudT::Ptr cloud_ptr(new PointCloudT);
    for(SubmapObj& submap_i: submaps_gt){
        *cloud_ptr = submap_i.submap_pcl_;
        pcl::io::savePCDFileASCII(folder_path.string() + "/submap_" + std::to_string(i) + ".pcd",
                                  submap_i.submap_pcl_);
        rgbVis(viewer, cloud_ptr, i);
        i++;
    }

    std::cout << "Number of submaps " << submaps_gt.size() << std::endl;

    // Detect overlaps between submaps
    SubmapsVec submaps_prev, submaps_reg;
    ofstream fileOutputStream;
    fileOutputStream.open("loop_closures.txt", std::ofstream::out);

    double info_thres = 0.1;
    for(SubmapObj& submap_i: submaps_gt){
        std::cout << " Submap " << submap_i.submap_id_ << std::endl;

        // Look for loop closures
        for(SubmapObj& submap_k: submaps_reg){
            // Don't look for overlaps between consecutive ones
            if(submap_k.submap_id_ != submap_i.submap_id_ - 1){
                submaps_prev.push_back(submap_k);
            }
        }
        // Submaps in map_frame?
        bool submaps_in_map_tf = true;
        submap_i.findOverlaps(submaps_in_map_tf, overlap_area, submaps_prev);
        submaps_prev.clear();

        // If potential loop closure detected
        if(!submap_i.overlaps_idx_.empty()){
            // Save loop closure to txt
            if(fileOutputStream.is_open()){
                fileOutputStream << submap_i.submap_id_;
                for(unsigned int j=0; j<submap_i.overlaps_idx_.size(); j++){
                    fileOutputStream << " " << submap_i.overlaps_idx_.at(j);
                }
                fileOutputStream << "\n";
            }
        }
        // Nacho: this can be done better to save mem
        submaps_reg.push_back(submap_i);    // Add registered submap_i
    }
    fileOutputStream.close();

    // // Benchmark GT
    // benchmark::track_error_benchmark benchmark("real_data");
    // PointsT gt_map = pclToMatrixSubmap(submaps_gt);
    // PointsT gt_track = trackToMatrixSubmap(submaps_gt);
    // benchmark.add_ground_truth(gt_map, gt_track);
    // // ceres::optimizer::saveOriginalTrajectory(submaps_gt); // Save original trajectory to txt
    // std::cout << "Visualizing original survey, press q to continue" << std::endl;

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    return 0;
}
