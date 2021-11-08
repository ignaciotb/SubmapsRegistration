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

void kmeansFPFH(boost::shared_ptr<Kmeans>& k_means,
                PointCloud<FPFHSignature33>::Ptr features, 
                YAML::Node config)
{

    // K-means clustering on SHOT descriptors
    k_means.reset(new Kmeans(static_cast<int>(features->size()), 33));
    int k_clusters = config["k_means_clusters"].as<double>();
    k_means->setClusterSize(k_clusters);
    // add points to the clustering
    for (const auto &point : features->points)
    {
        std::vector<float> data(33);
        for (int idx = 0; idx < 33; idx++)
            data[idx] = point.histogram[idx];
        k_means->addDataPoint(data);
    }

    // k-means clustering
    k_means->kMeans();
}

void kmeansSHOT(boost::shared_ptr<Kmeans>& k_means, 
                PointCloud<SHOT352>::Ptr features, 
                YAML::Node config)
{

    // K-means clustering on SHOT descriptors
    k_means.reset(new Kmeans(static_cast<int>(features->size()), 352));
    int k_clusters = config["k_means_clusters"].as<double>();
    k_means->setClusterSize(k_clusters);
    // add points to the clustering
    for (const auto &point : features->points)
    {
        std::vector<float> data(352);
        for (int idx = 0; idx < 352; idx++)
            data[idx] = point.descriptor[idx];
        k_means->addDataPoint(data);
    }

    // k-means clustering
    k_means->kMeans();
}

void kmeansKeypoints(boost::shared_ptr<Kmeans> &k_means, 
                     PointCloudT::Ptr keypoints, 
                     YAML::Node config)
{
    // K-means clustering on keypoints
    k_means.reset(new Kmeans(static_cast<int>(keypoints->size()), 3));
    int k_clusters = config["k_means_clusters"].as<double>();
    k_means->setClusterSize(k_clusters);
    // add points to the clustering
    for (const auto &point : keypoints->points)
    {
        std::vector<float> data(3);
        int idx = 0;
        data[idx] = point.x;
        data[idx + 1] = point.y;
        data[idx + 2] = point.z;
        k_means->addDataPoint(data);
    }

    // k-means clustering
    k_means->kMeans();
}

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

    // Visualize downsampled pointcloud
    rgbVis(viewer, cloud_ptr, 0);
    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    // Extract keypoints
    auto t1 = high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1(new pcl::PointCloud<pcl::PointXYZ>);
    if(config["harris"].as<bool>()){
        std::cout << "Extracting Harris keypoints" << std::endl;
        harrisKeypoints(cloud_ptr, *keypoints_1, config);
    }
    else if (config["sift"].as<bool>()){
        std::cout << "Extracting SIFT keypoints" << std::endl;
        siftKeypoints(cloud_ptr, *keypoints_1, config);
    }
    else{
        std::cerr << "Choose an implemented keypoint extraction method" << std::endl;
    }
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

    // Compute features and kmeans clustering
    boost::shared_ptr<Kmeans> k_means;

    if(config["shot"].as<bool>()){
        std::cout << "Computing SHOT features" << std::endl;
        PointCloud<SHOT352>::Ptr features_1(new PointCloud<SHOT352>);
        estimateSHOT(keypoints_1, features_1, config);
        
        std::cout << "Kmeans clustering of SHOT descriptors" << std::endl;
        kmeansSHOT(k_means, features_1, config);
    }
    else if (config["fpfh"].as<bool>()){
        std::cout << "Computing FPFH features" << std::endl;
        PointCloud<FPFHSignature33>::Ptr features_1(new PointCloud<FPFHSignature33>);
        estimateFPFH(keypoints_1, features_1, config);
        
        std::cout << "Kmeans clustering of FPFH descriptors" << std::endl;
        kmeansFPFH(k_means, features_1, config);
    }
    else{
        std::cout << "Kmeans clustering of keypoints" << std::endl;
        kmeansKeypoints(k_means, keypoints_1, config);
    }

    // NACHO: kmeans.h has been modified locally to add the accessor get_clustersToPoints()
    pcl::Kmeans::ClustersToPoints clusters2points = k_means->get_clustersToPoints();
    std::cout << "Number of clusters " << clusters2points.size() << std::endl;

    viewer->removePointCloud("keypoints_src", v1);
    PointCloud<PointT>::Ptr pcl_clusters(new PointCloud<PointT>);
    for(int i=0; i<clusters2points.size(); i++){
        pcl_clusters->points.clear();
        for (const auto &pid : clusters2points[i])
        {
            PointT p = keypoints_1->points[pid];
            pcl_clusters->points.push_back(p);
        }
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clusters_color(pcl_clusters,
                                                                            rand() / 256., rand() / 256., rand() / 256.);
        viewer->addPointCloud(pcl_clusters, clusters_color, "clusters_src_" + std::to_string(i), v1);
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