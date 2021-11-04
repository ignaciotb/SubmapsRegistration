#include <feature_matching/corresp_matching.hpp>
#include <feature_matching/utils_visualization.hpp>

void extractKeypointsCorrespondences(const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_2,
                                     CorrespondencesPtr good_correspondences)
{
  // Basic correspondence estimation between keypoints
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
  CorrespondencesPtr all_correspondences(new Correspondences);
  est.setInputTarget(keypoints_1);
  est.setInputSource(keypoints_2);
  est.determineReciprocalCorrespondences(*all_correspondences, 30.0);
  rejectBadCorrespondences(all_correspondences, keypoints_1, keypoints_2, *good_correspondences);

  std::cout << "Number of correspondances " << all_correspondences->size() << std::endl;
  std::cout << "Number of good correspondances " << good_correspondences->size() << std::endl;
}

void extractFeaturesCorrespondences(const PointCloud<SHOT352>::Ptr &shot_src,
                                    const PointCloud<SHOT352>::Ptr &shot_trg,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_2,
                                    CorrespondencesPtr good_correspondences)
{
  // Basic correspondence estimation between keypoints
  pcl::registration::CorrespondenceEstimation<SHOT352, SHOT352> est;
  CorrespondencesPtr all_correspondences(new Correspondences);
  est.setInputTarget(shot_src);
  est.setInputSource(shot_trg);
  est.determineReciprocalCorrespondences(*all_correspondences, 40.0);
  rejectBadCorrespondences(all_correspondences, keypoints_1, keypoints_2, *good_correspondences);

  std::cout << "Number of correspondances " << all_correspondences->size() << std::endl;
  std::cout << "Number of good correspondances " << good_correspondences->size() << std::endl;
}

int main(int, char **argv)
{
    // Parse the command line arguments for .pcd files
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1_noisy(new pcl::PointCloud<pcl::PointXYZ>);

    // Load the files
    if (pcl::io::loadPCDFile (argv[1], *cloud_1) < 0){
        PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
        return (-1);
    }

    // if (pcl::io::loadPCDFile (argv[2], *cloud_trg) < 0){
    //     PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
    //     return (-1);
    // }

    // Initial noisy misalignment between pointclouds
    std::random_device rd{};
    std::mt19937 seed{rd()};
    
    double tf_std_dev = 0.6;
    std::normal_distribution<double> d2{0, tf_std_dev};
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    double theta = M_PI / 10. + d2(seed);
    transformation_matrix (0, 0) = cos (theta);
    transformation_matrix (0, 1) = -sin (theta);
    transformation_matrix (1, 0) = sin (theta);
    transformation_matrix (1, 1) = cos (theta);
    transformation_matrix (0, 3) = -10. + d2(seed);
    transformation_matrix (1, 3) = -10. + d2(seed);
    transformation_matrix (2, 3) = 0.0;
    pcl::transformPointCloud(*cloud_1, *cloud_1_noisy, transformation_matrix);

    // Gaussian noise to points in input clouds
    double pcl_std_dev = 0.01;
    std::normal_distribution<double> d{0, pcl_std_dev};
    for (unsigned int i = 0; i < cloud_1->points.size(); i++)
    {
      cloud_1->points.at(i).x = cloud_1->points.at(i).x + d(seed);
      cloud_1->points.at(i).y = cloud_1->points.at(i).y + d(seed);
      cloud_1->points.at(i).z = cloud_1->points.at(i).z + d(seed);

      cloud_1_noisy->points.at(i).x = cloud_1_noisy->points.at(i).x + d(seed);
      cloud_1_noisy->points.at(i).y = cloud_1_noisy->points.at(i).y + d(seed);
      cloud_1_noisy->points.at(i).z = cloud_1_noisy->points.at(i).z + d(seed);
    }

    // Visualize initial point clouds
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    rgbVis(viewer, cloud_1, 0);
    rgbVis(viewer, cloud_1_noisy, 1);

    while (!viewer->wasStopped())
    {
      viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    // Extract keypoints
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_2(new pcl::PointCloud<pcl::PointXYZ>);
    harrisKeypoints(cloud_1, *keypoints_1);
    harrisKeypoints(cloud_1_noisy, *keypoints_2);

    // Visualize keypoints
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints1_color_handler(keypoints_1, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints2_color_handler(keypoints_2, 0, 255, 0);
    viewer->addPointCloud(keypoints_1, keypoints1_color_handler, "keypoints_src", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_src");
    viewer->addPointCloud(keypoints_2, keypoints2_color_handler, "keypoints_trg", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints_trg");

    // Compute SHOT descriptors
    PointCloud<SHOT352>::Ptr shot_1(new PointCloud<SHOT352>);
    PointCloud<SHOT352>::Ptr shot_2(new PointCloud<SHOT352>);
    estimateSHOT(keypoints_1, shot_1);
    estimateSHOT(keypoints_2, shot_2);

    // Extract correspondences between keypoints/features
    CorrespondencesPtr good_correspondences(new Correspondences);
    // extractKeypointsCorrespondences(keypoints_1, keypoints_2, good_correspondences);
    extractFeaturesCorrespondences(shot_1, shot_2, keypoints_1, keypoints_2, good_correspondences);

    // Extract correspondences between keypoints
    plotCorrespondences(*viewer, *good_correspondences, keypoints_1, keypoints_2);
    while (!viewer->wasStopped())
    {
      viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    // Best transformation between the two sets of keypoints given the remaining correspondences
    Eigen::Matrix4f transform;
    TransformationEstimationSVD<PointXYZ, PointXYZ> trans_est;
    trans_est.estimateRigidTransformation(*keypoints_1, *keypoints_2, *good_correspondences, transform);
    pcl::transformPointCloud(*cloud_1_noisy, *cloud_1_noisy, transform.inverse());

    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
    rgbVis(viewer, cloud_1, 0);
    rgbVis(viewer, cloud_1_noisy, 1);
    while (!viewer->wasStopped())
    {
      viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    // Run GICP
    runGicp(cloud_1_noisy, cloud_1);

    viewer->removeAllPointClouds();
    rgbVis(viewer, cloud_1, 0);
    rgbVis(viewer, cloud_1_noisy, 1);
    while (!viewer->wasStopped())
    {
      viewer->spinOnce();
    }
    viewer->resetStoppedFlag();

    return 0;
}

