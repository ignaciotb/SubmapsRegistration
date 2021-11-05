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

#include <pcl/registration/warp_point_rigid.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/gicp.h>

#include <pcl/conversions.h>
#include <pcl/filters/uniform_sampling.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/feature.h>
#include <pcl/features/shot_lrf.h>
#include "yaml-cpp/yaml.h"

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace pcl::registration;
using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace pcl
{
    template <>
    struct SIFTKeypointFieldSelector<PointXYZ>
    {
        inline float
        operator()(const PointXYZ &p) const
        {
            return p.z;
        }
    };
}

////////////////////////////////////////////////////////////////////////////////
void siftKeypoints(const PointCloudT::Ptr cloud_in,
                   PointCloud<PointXYZ> &keypoints_src)
{

    // Parameters for sift computation
    const float min_scale = 1.f;
    const int n_octaves = 6;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.005f;

    // Estimate the sift interest points using z values from xyz as the Intensity variants
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> detector;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    detector.setSearchMethod(tree);
    detector.setScales(min_scale, n_octaves, n_scales_per_octave);
    detector.setMinimumContrast(min_contrast);
    detector.setInputCloud(cloud_in);
    detector.compute(result);
    pcl::console::print_highlight("Detected %zd points \n", result.size());
    pcl::PointIndicesConstPtr keypoints_indices = detector.getKeypointsIndices();
    // std::cout << "Indices detected" << keypoints_indices. << std::endl;
    copyPointCloud(result, keypoints_src);

    // TODO: fix keypoints positions when coming out of detector.compute()
    // for (const int id : keypoints_indices->indices)
    // {
    //     pcl::PointXYZ point;
    //     point.x = cloud_in->points[id].x;
    //     point.y = cloud_in->points[id].y;
    //     point.z = cloud_in->points[id].z;
    //     keypoints_src.push_back(point);
    // }

    std::cout << "No of SIFT points in the result are " << keypoints_src.size() << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void harrisKeypoints(const PointCloud<PointXYZ>::Ptr &src,
                     PointCloud<PointXYZ> &keypoints_src, YAML::Node config)
{
    //YAML::Node config = YAML::LoadFile("./config.yaml");
    // cout << config["harris_kps_radius"] << endl;
    // Compute normals
    //double radius = 5.;
    double radius = config["harris_kps_radius"].as<double>();
    double thres = config["harris_kps_thres"].as<double>();
    NormalEstimation<PointXYZ, Normal> normal_est;
    PointCloud<pcl::Normal>::Ptr normals(new PointCloud<pcl::Normal>());
    normal_est.setInputCloud(src);
    normal_est.setRadiusSearch(radius);
    normal_est.compute(*normals);

    // Create src with intensity
    PointCloud<PointXYZI>::Ptr src_i(new PointCloud<PointXYZI>());
    src_i->sensor_origin_ = src->sensor_origin_;
    src_i->sensor_orientation_ = src->sensor_orientation_;
    for (PointT &point : src->points)
    {
        pcl::PointXYZI pointI;
        pointI.x = point.x;
        pointI.y = point.y;
        pointI.z = point.z;
        src_i->points.push_back(pointI);
    }

    PointCloud<PointXYZI> keypoints_src_i;
    pcl::HarrisKeypoint3D<pcl::PointXYZI, pcl::PointXYZI, Normal> detector;
    detector.setNonMaxSupression(true);
    detector.setInputCloud(src_i);
    detector.setRadius(0.1);
    detector.setNormals(normals);
    detector.setThreshold(thres);
    detector.compute(keypoints_src_i);
    pcl::console::print_highlight("Normals %zd \n", normals->size());
    pcl::console::print_highlight("Detected %zd points \n", keypoints_src_i.size());
    pcl::PointIndicesConstPtr keypoints_indices = detector.getKeypointsIndices();

    for (const int id : keypoints_indices->indices)
    {
        pcl::PointXYZ point;
        point.x = src_i->points[id].x;
        point.y = src_i->points[id].y;
        point.z = src_i->points[id].z;
        keypoints_src.push_back(point);
    }
}

////////////////////////////////////////////////////////////////////////////////
void uniformKeypoints(const PointCloud<PointXYZ>::Ptr &src,
                       const PointCloud<PointXYZ>::Ptr &tgt,
                       PointCloud<PointXYZ> &keypoints_src,
                       PointCloud<PointXYZ> &keypoints_tgt)
{
    // Get an uniform grid of keypoints
    UniformSampling<PointXYZ> uniform;
    uniform.setRadiusSearch(1); // 1m

    uniform.setInputCloud(src);
    uniform.filter(keypoints_src);

    uniform.setInputCloud(tgt);
    uniform.filter(keypoints_tgt);

    // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
    // pcl_viewer source_pcd keypoints_src.pcd -ps 1 -ps 10
    savePCDFileBinary("keypoints_src.pcd", keypoints_src);
    savePCDFileBinary("keypoints_tgt.pcd", keypoints_tgt);
}

////////////////////////////////////////////////////////////////////////////////
void estimateSHOT(const PointCloud<PointXYZ>::Ptr &keypoints,
                  PointCloud<SHOT352>::Ptr shot_src)
{
    // Compute normals
    double radius = 150.0; 
    NormalEstimation<PointXYZ, Normal> normal_est;
    PointCloud<pcl::Normal>::Ptr normals(new PointCloud<pcl::Normal>());
    normal_est.setInputCloud(keypoints);
    normal_est.setRadiusSearch(1);
    normal_est.compute(*normals);

    // // Compute SHOT descriptors
    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shotEstimation;
    shotEstimation.setInputCloud(keypoints);
    shotEstimation.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    shotEstimation.setSearchMethod(tree);
    shotEstimation.setRadiusSearch(2);
    shotEstimation.setKSearch(0);
    
    // Compute reference frames externally
    PointCloud<ReferenceFrame>::Ptr frames(new PointCloud<ReferenceFrame>());
    SHOTLocalReferenceFrameEstimation<PointT, pcl::ReferenceFrame> lrf_estimator;
    lrf_estimator.setRadiusSearch(1);
    lrf_estimator.setInputCloud(keypoints);
    // lrf_estimator.setIndices(indices2);
    // lrf_estimator.setSearchSurface(points);
    lrf_estimator.compute(*frames);

    shotEstimation.setInputReferenceFrames(frames);

    // Actually compute the spin images
    shotEstimation.compute(*shot_src);
    std::cout << "SHOT output size: " << shot_src->points.size() << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void estimateFPFH(const PointCloud<PointXYZ>::Ptr &src,
                  const PointCloud<PointXYZ>::Ptr &tgt,
                  const PointCloud<Normal>::Ptr &normals_src,
                  const PointCloud<Normal>::Ptr &normals_tgt,
                  const PointCloud<PointXYZ>::Ptr &keypoints_src,
                  const PointCloud<PointXYZ>::Ptr &keypoints_tgt,
                  PointCloud<FPFHSignature33> &fpfhs_src,
                  PointCloud<FPFHSignature33> &fpfhs_tgt)
{
    FPFHEstimation<PointXYZ, Normal, FPFHSignature33> fpfh_est;
    fpfh_est.setInputCloud(keypoints_src);
    fpfh_est.setInputNormals(normals_src);
    fpfh_est.setRadiusSearch(1); // 1m
    fpfh_est.setSearchSurface(src);
    fpfh_est.compute(fpfhs_src);

    fpfh_est.setInputCloud(keypoints_tgt);
    fpfh_est.setInputNormals(normals_tgt);
    fpfh_est.setSearchSurface(tgt);
    fpfh_est.compute(fpfhs_tgt);

    // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
    // pcl_viewer fpfhs_src.pcd
    // PCLPointCloud2 s, t, out;
    // toPCLPointCloud2(*keypoints_src, s);
    // toPCLPointCloud2(fpfhs_src, t);
    // concatenateFields(s, t, out);
    // savePCDFile("fpfhs_src.pcd", out);
    // toPCLPointCloud2(*keypoints_tgt, s);
    // toPCLPointCloud2(fpfhs_tgt, t);
    // concatenateFields(s, t, out);
    // savePCDFile("fpfhs_tgt.pcd", out);
}

////////////////////////////////////////////////////////////////////////////////
void estimateNormals(const PointCloud<PointXYZ>::Ptr &src,
                     const PointCloud<PointXYZ>::Ptr &tgt,
                     PointCloud<Normal> &normals_src,
                     PointCloud<Normal> &normals_tgt)
{
    NormalEstimation<PointXYZ, Normal> normal_est;
    normal_est.setInputCloud(src);
    normal_est.setRadiusSearch(5); // 50cm
    normal_est.compute(normals_src);

    normal_est.setInputCloud(tgt);
    normal_est.compute(normals_tgt);

    // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
    // pcl_viewer normals_src.pcd
    PointCloud<PointNormal> s, t;
    copyPointCloud(*src, s);
    copyPointCloud(normals_src, s);
    copyPointCloud(*tgt, t);
    copyPointCloud(normals_tgt, t);
    savePCDFileBinary("normals_src.pcd", s);
    savePCDFileBinary("normals_tgt.pcd", t);
}

////////////////////////////////////////////////////////////////////////////////
void findCorrespondences(const PointCloud<FPFHSignature33>::Ptr &fpfhs_src,
                         const PointCloud<FPFHSignature33>::Ptr &fpfhs_tgt,
                         Correspondences &all_correspondences)
{
    CorrespondenceEstimation<FPFHSignature33, FPFHSignature33> est;
    est.setInputCloud(fpfhs_src);
    est.setInputTarget(fpfhs_tgt);
    // est.determineReciprocalCorrespondences(all_correspondences);
    est.determineCorrespondences(all_correspondences, 100);
}

////////////////////////////////////////////////////////////////////////////////
void rejectBadCorrespondences(const CorrespondencesPtr &all_correspondences,
                              const PointCloud<PointXYZ>::Ptr &keypoints_src,
                              const PointCloud<PointXYZ>::Ptr &keypoints_tgt,
                              Correspondences &remaining_correspondences)
{
    CorrespondenceRejectorDistance rej;
    rej.setInputSource<PointXYZ>(keypoints_src);
    rej.setInputTarget<PointXYZ>(keypoints_tgt);
    rej.setMaximumDistance(40); // m
    rej.setInputCorrespondences(all_correspondences);
    rej.getCorrespondences(remaining_correspondences);
}

////////////////////////////////////////////////////////////////////////////////
void computeTransformation(const PointCloud<PointXYZ>::Ptr &src,
                           const PointCloud<PointXYZ>::Ptr &tgt,
                           Eigen::Matrix4f &transform,
                           CorrespondencesPtr &result_correspondences)
{
    // Get an uniform grid of keypoints
    PointCloud<PointXYZ>::Ptr keypoints_src(new PointCloud<PointXYZ>),
        keypoints_tgt(new PointCloud<PointXYZ>);

    uniformKeypoints(src, tgt, *keypoints_src, *keypoints_tgt);
    print_info("Found %zu and %zu keypoints for the source and target datasets.\n", static_cast<std::size_t>(keypoints_src->size()), static_cast<std::size_t>(keypoints_tgt->size()));

    // Compute normals for all points keypoint
    PointCloud<Normal>::Ptr normals_src(new PointCloud<Normal>),
        normals_tgt(new PointCloud<Normal>);
    estimateNormals(src, tgt, *normals_src, *normals_tgt);
    print_info("Estimated %zu and %zu normals for the source and target datasets.\n", static_cast<std::size_t>(normals_src->size()), static_cast<std::size_t>(normals_tgt->size()));

    // Compute FPFH features at each keypoint
    PointCloud<FPFHSignature33>::Ptr fpfhs_src(new PointCloud<FPFHSignature33>),
        fpfhs_tgt(new PointCloud<FPFHSignature33>);
    estimateFPFH(src, tgt, normals_src, normals_tgt, keypoints_src, keypoints_tgt, *fpfhs_src, *fpfhs_tgt);

    // Find correspondences between keypoints in FPFH space
    CorrespondencesPtr all_correspondences(new Correspondences),
        good_correspondences(new Correspondences);
    findCorrespondences(fpfhs_src, fpfhs_tgt, *all_correspondences);

    // Reject correspondences based on their XYZ distance
    rejectBadCorrespondences(all_correspondences, keypoints_src, keypoints_tgt, *good_correspondences);

    // Keep only best ones?
    //   sort(corrs->begin(), corrs->end(), pcl::isBetterCorrespondence);
    //   reverse(corrs->begin(), corrs->end());
    result_correspondences.reset(new pcl::Correspondences(*good_correspondences));

    std::cout << "Number of correspondances " << all_correspondences->size() << std::endl;
    std::cout << "Number of good correspondances " << good_correspondences->size() << std::endl;
    // for (const auto& corr : (*good_correspondences))
    //   std::cerr << corr << std::endl;
    // Obtain the best transformation between the two sets of keypoints given the remaining correspondences
    TransformationEstimationSVD<PointXYZ, PointXYZ> trans_est;
    trans_est.estimateRigidTransformation(*keypoints_src, *keypoints_tgt, *all_correspondences, transform);
}


void runGicp(PointCloudT::Ptr &src_cloud, const PointCloudT::Ptr &trg_cloud)
{

    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;

    // Constrain GICP to x,y, yaw
    pcl::registration::WarpPointRigid3D<PointT, PointT>::Ptr warp_fcn(new pcl::registration::WarpPointRigid3D<PointT, PointT>);

    pcl::registration::TransformationEstimationLM<PointT, PointT>::Ptr te(new pcl::registration::TransformationEstimationLM<PointT, PointT>);
    te->setWarpFunction(warp_fcn);
    gicp.setTransformationEstimation(te);

    gicp.setInputSource(src_cloud);
    gicp.setInputTarget(trg_cloud);

    gicp.setMaxCorrespondenceDistance(10);
    gicp.setMaximumIterations(200);
    // gicp.setMaximumOptimizerIterations(200);
    // gicp.setRANSACIterations(100);
    // gicp.setRANSACOutlierRejectionThreshold(10);
    gicp.setTransformationEpsilon(1e-4);
    // gicp.setUseReciprocalCorrespondences(true);

    gicp.align(*src_cloud);
}