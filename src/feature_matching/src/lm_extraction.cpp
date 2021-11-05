#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>

#include "feature_matching/cxxopts.hpp"
#include "feature_matching/utils_visualization.hpp"
#include <feature_matching/corresp_matching.hpp>

#include <pcl/io/auto_io.h>

#include <pcl/ml/kmeans.h>

#include "yaml-cpp/parser.h"
#include "yaml-cpp/node/detail/node_data.h"

using namespace Eigen;
using namespace std;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

int main(int argc, char **argv)
{

    // Inputs
    std::string folder_str, input_path, yaml_file;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()("help", "Print help")
    ("input_map", "PCD map", cxxopts::value(input_path))
    ("yaml_file", "PCD map", cxxopts::value(yaml_file));

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    // Load the yaml file
    boost::filesystem::path yaml_path(yaml_file);
    YAML::Node config = YAML::LoadFile(yaml_path.string());

    // Parse submaps from cereal file
    boost::filesystem::path map_path(input_path);
    std::cout << "Input file " << map_path.string() << std::endl;

    // Visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    // Load map
    PointCloudT::Ptr cloud_ptr(new PointCloudT);

    if (pcl::io::loadPCDFile(map_path.string(), *cloud_ptr) < 0)
    {
        PCL_ERROR("Error loading cloud %s.\n", map_path.string());
        return (-1);
    }

    // Get an uniform grid of keypoints
    pcl::console::print_highlight("Before sampling %zd points \n", cloud_ptr->size());
    UniformSampling<PointXYZ> uniform;
    uniform.setRadiusSearch(1); // m
    uniform.setInputCloud(cloud_ptr);
    uniform.filter(*cloud_ptr);
    pcl::console::print_highlight("After sampling %zd points \n", cloud_ptr->size());

    rgbVis(viewer, cloud_ptr, 0);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    // Extract keypoints
    auto t1 = high_resolution_clock::now();
    std::cout << "Extracting keypoints" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1(new pcl::PointCloud<pcl::PointXYZ>);
    harrisKeypoints(cloud_ptr, *keypoints_1, config);
    // siftKeypoints(cloud_ptr, *keypoints_1, config);

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    // Visualize keypoints
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints1_color(keypoints_1, 255, 0, 0);
    viewer->addPointCloud(keypoints_1, keypoints1_color, "keypoints_src", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_src");

    std::cout << "Keypoint extraction duration (s) " << ms_double.count()/1000. << std::endl;

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    // Compute SHOT and save
    std::cout << "Computing features" << std::endl;
    PointCloud<SHOT352>::Ptr features_1(new PointCloud<SHOT352>);
    estimateSHOT(keypoints_1, features_1, config);

    // std::cout << "Computing FPFH features" << std::endl;
    // PointCloud<SHOT352>::Ptr features_1(new PointCloud<FPFHSignature33>);
    // estimateFPFH(keypoints_1, features_1, config);

    // pcl::io::save(submaps_path.string() + "/shot_map.bin",
    //                           *features_1);

    // K-means clustering
    std::cout << "Kmeans clustering" << std::endl;
    Kmeans k_means(static_cast<int>(features_1->size()), 352);
    int k_clusters = config["k_means_clusters"].as<double>();
    k_means.setClusterSize(k_clusters);
    // add points to the clustering
    for (const auto &point : features_1->points)
    {
        std::vector<float> data(352);
        for (int idx = 0; idx < 352; idx++)
            data[idx] = point.descriptor[idx];
        k_means.addDataPoint(data);
    }

    // k-means clustering
    k_means.kMeans();

    // NACHO: kmeans.h has been modified locally to add the accessor get_clustersToPoints()
    pcl::Kmeans::ClustersToPoints clusters2points = k_means.get_clustersToPoints();
    std::cout << "Clusters " << clusters2points.size() << std::endl;

    viewer->removePointCloud("keypoints_src", v1);
    PointCloud<PointT>::Ptr shot_clusters(new PointCloud<PointT>);
    for(int i=0; i<clusters2points.size(); i++){
        shot_clusters->points.clear();
        for (const auto &pid : clusters2points[i])
        {
            PointT p = keypoints_1->points[pid];
            shot_clusters->points.push_back(p);
        }
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clusters_color(shot_clusters,
                                                                            rand() / 256., rand() / 256., rand() / 256.);
        viewer->addPointCloud(shot_clusters, clusters_color, "clusters_src_" + std::to_string(i), v1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7,
                                                    "clusters_src_"+std::to_string(i));
    }

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    return 0;
}