#include <iostream>
#include <string>
#include <random>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/conversions.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
// using namespace pcl::registration;
using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<PointT> PointCloudT;


int v1(0);
int v2(1);


std::tuple<uint8_t, uint8_t, uint8_t> jet(double x)
{
    const double rone = 0.8;
    const double gone = 1.0;
    const double bone = 1.0;
    double r, g, b;

    x = (x < 0 ? 0 : (x > 1 ? 1 : x));

    if (x < 1. / 8.)
    {
        r = 0;
        g = 0;
        b = bone * (0.5 + (x) / (1. / 8.) * 0.5);
    }
    else if (x < 3. / 8.)
    {
        r = 0;
        g = gone * (x - 1. / 8.) / (3. / 8. - 1. / 8.);
        b = bone;
    }
    else if (x < 5. / 8.)
    {
        r = rone * (x - 3. / 8.) / (5. / 8. - 3. / 8.);
        g = gone;
        b = (bone - (x - 3. / 8.) / (5. / 8. - 3. / 8.));
    }
    else if (x < 7. / 8.)
    {
        r = rone;
        g = (gone - (x - 5. / 8.) / (7. / 8. - 5. / 8.));
        b = 0;
    }
    else
    {
        r = (rone - (x - 7. / 8.) / (1. - 7. / 8.) * 0.5);
        g = 0;
        b = 0;
    }

    return std::make_tuple(uint8_t(255. * r), uint8_t(255. * g), uint8_t(255. * b));
}

bool next_iteration_icp = false;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *nothing)
{
    if (event.getKeySym() == "space" && event.keyDown())
        next_iteration_icp = true;
}

void plotCorrespondences(pcl::visualization::PCLVisualizer &viewer,
                         pcl::Correspondences &corrs,
                         PointCloudT::Ptr &src,
                         PointCloudT::Ptr &trg)
{

    int j = 0;
    Eigen::Vector3i dr_color = Eigen::Vector3i(rand() % 256, rand() % 256, rand() % 256);
    for (auto corr_i : corrs)
    {
        viewer.addLine(src->at(corr_i.index_match), trg->at(corr_i.index_query),
                       dr_color[0], dr_color[1], dr_color[2], "corr_" + std::to_string(j));
        j++;
    }
    viewer.spinOnce();
}


void pclVisualizer(pcl::visualization::PCLVisualizer &viewer,
                   const PointCloudT::Ptr cloud_in,
                   const PointCloudT::Ptr cloud_tr,
                   const PointCloudT::Ptr cloud_icp)
{

    // Viewports
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    float bckgr_gray_level = 0.0; // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    // Original point cloud is white
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_in, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl,
                                                                              (int)255 * txt_gray_lvl);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h(cloud_tr, 20, 180, 20);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h(cloud_icp, 180, 20, 20);

    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_src_v1", v1);
    viewer.addPointCloud(cloud_tr, cloud_tr_color_h, "cloud_trg_v1", v1);
    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_src_v2", v2);
    viewer.addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_trg_v2", v2);

    // Text descriptions and background
    viewer.addText("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
    viewer.addText("White: Original point cloud\nRed: PICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);
    viewer.setBackgroundColor(txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, v1);
    viewer.setBackgroundColor(txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, v2);

    // Set camera position and orientation
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024); // Visualiser window size
    viewer.registerKeyboardCallback(&keyboardEventOccurred, (void *)NULL);
}


void rgbVis(pcl::visualization::PCLVisualizer::Ptr &viewer,
            pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
            int i)
{
    int vp1_;

    // pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    float black = 0.0; // Black
    float white = 1.0 - black;
    // viewer->createViewPort (0.0, 0.0, 1.0, 1.0, vp1_);

    // unsigned int i = 0;
    PointCloudRGB::Ptr cloud_clr(new PointCloudRGB);
    // Find max and min depth in map
    PointT min, max;
    pcl::getMinMax3D(*cloud_in, min, max);
    std::cout << "Max " << max.getArray3fMap().transpose() << std::endl;
    std::cout << "Min " << min.getArray3fMap().transpose() << std::endl;
    // Normalize and give colors based on z
    for (PointT &pointt : cloud_in->points)
    {
        pcl::PointXYZRGB pointrgb;
        pointrgb.x = pointt.x;
        pointrgb.y = pointt.y;
        pointrgb.z = pointt.z;
        std::tuple<uint8_t, uint8_t, uint8_t> colors_rgb;
        colors_rgb = jet((pointt.z - min.z) / (max.z - min.z));
        std::uint32_t rgb = (static_cast<std::uint32_t>(std::get<0>(colors_rgb)) << 16 |
                             static_cast<std::uint32_t>(std::get<1>(colors_rgb)) << 8 |
                             static_cast<std::uint32_t>(std::get<2>(colors_rgb)));
        pointrgb.rgb = *reinterpret_cast<float *>(&rgb);
        cloud_clr->points.push_back(pointrgb);
    }
    std::cout << cloud_clr->points.size() << std::endl;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_h(cloud_clr);
    viewer->addPointCloud(cloud_clr, rgb_h, "submap_" + std::to_string(i));
    // viewer->addCoordinateSystem(3.0, submap.submap_tf_, "gt_cloud_" + std::to_string(i), vp1_);
    // viewer->addCoordinateSystem(5.0, cloud_in->sensor_origin_(0), cloud_in->sensor_origin_(1),
    //                             cloud_in->sensor_origin_(2), "submap_" + std::to_string(i));

    // return (viewer);
}