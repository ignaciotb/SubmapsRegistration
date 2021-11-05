#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>

#include "feature_matching/cxxopts.hpp"
#include "feature_matching/utils_visualization.hpp"
#include "yaml-cpp/yaml.h"

using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{

    // Inputs
    std::string folder_str, input_path, output_path;
    int first_submap, last_submap;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()("help", "Print help")
    ("input_folder", "Path to folder with pcd submaps", cxxopts::value(input_path))
    ("first_submap", "Index of first submap from folder to visualize", cxxopts::value(first_submap))
    ("last_submap", "Index of last submap from folder to visualize", cxxopts::value(last_submap));

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    // Parse submaps from cereal file
    boost::filesystem::path submaps_path(input_path);
    std::cout << "Input folder " << submaps_path << std::endl;

    // Visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    // Save submaps to disk
    PointCloudT::Ptr cloud_ptr(new PointCloudT);
    for (int i = first_submap; i < last_submap; i++)
    {
        std::cout << "Submap " << i << std::endl;
        if (pcl::io::loadPCDFile(submaps_path.string() + "/submap_" + std::to_string(i) + ".pcd", *cloud_ptr) < 0)
        {
            PCL_ERROR("Error loading cloud %s.\n", submaps_path.string() + "/submap_" + std::to_string(i) + ".pcd");
            return (-1);
        }
        rgbVis(viewer, cloud_ptr, i);

        while (!viewer->wasStopped())
        {
            viewer->spinOnce();
        }
        viewer->resetStoppedFlag();
    }


    return 0;
}
