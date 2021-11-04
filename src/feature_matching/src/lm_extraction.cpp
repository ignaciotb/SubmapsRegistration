#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>

#include "feature_matching/cxxopts.hpp"
#include "feature_matching/utils_visualization.hpp"
#include <feature_matching/corresp_matching.hpp>

#include <chrono>

using namespace Eigen;
using namespace std;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

int main(int argc, char **argv)
{

    // Inputs
    std::string folder_str, input_path, output_path;
    int first_submap, last_submap;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()("help", "Print help")("input_folder", "Path to folder with pcd submaps", cxxopts::value(input_path))("first_submap", "Index of first submap from folder to visualize", cxxopts::value(first_submap))("last_submap", "Index of last submap from folder to visualize", cxxopts::value(last_submap));

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

        // Get an uniform grid of keypoints
        pcl::console::print_highlight("Before sampling %zd points \n", cloud_ptr->size());
        UniformSampling<PointXYZ> uniform;
        uniform.setRadiusSearch(1); // m
        uniform.setInputCloud(cloud_ptr);
        uniform.filter(*cloud_ptr);
        pcl::console::print_highlight("After sampling %zd points \n", cloud_ptr->size());

        rgbVis(viewer, cloud_ptr, i);

        while (!viewer->wasStopped())
        {
            viewer->spinOnce();
        }
        viewer->resetStoppedFlag();
    }

    // Extract keypoints
    auto t1 = high_resolution_clock::now();
    std::cout << "Extracting keypoints" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1(new pcl::PointCloud<pcl::PointXYZ>);
    harrisKeypoints(cloud_ptr, *keypoints_1);
    
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    // Visualize keypoints
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints1_color(keypoints_1, 255, 0, 0);
    viewer->addPointCloud(keypoints_1, keypoints1_color, "keypoints_src", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_src");

    std::cout << "Keypoint extraction duration (s) " << ms_double.count()/1000. << std::endl;

    pcl::io::savePCDFileASCII(submaps_path.string() + "/keypoints_map.pcd",
                              *keypoints_1);

    // Compute SHOT and save
    PointCloud<SHOT352>::Ptr shot_1(new PointCloud<SHOT352>);
    estimateSHOT(keypoints_1, shot_1);

    PCLPointCloud2 s, t, out;
    toPCLPointCloud2(*keypoints_1, s);
    toPCLPointCloud2(*shot_1, t);
    concatenateFields(s, t, out);
    savePCDFile(submaps_path.string() + "/shot_map.pcd", out);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    return 0;
}