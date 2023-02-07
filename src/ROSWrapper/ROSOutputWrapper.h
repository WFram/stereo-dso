#include <Output3DWrapper.h>
#include <memory>
#include <mutex>

#include "cv_bridge/cv_bridge.h"

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include <std_srvs/Empty.h>

namespace dso {
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZINormal> PointCloudXYZINormal;

class ROSOutputWrapper : public dso::IOWrap::Output3DWrapper {
 public:
  ROSOutputWrapper();

  virtual void publishInitSignal() override;

  virtual void publishCamPose(dso::FrameShell *frame, dso::CalibHessian *HCalib) override;

  virtual void publishKeyframes(std::vector<dso::FrameHessian *> &frames, bool final,
                                dso::CalibHessian *HCalib) override;

  virtual void pushLiveFrame(dso::FrameHessian *image) override;

  void publishOutput();

  void undistortCallback(const sensor_msgs::ImageConstPtr img);

  void publish_reference_cloud();

 private:
  ros::NodeHandle nh;
  ros::Publisher dsoOdomHighFreqPublisher, dsoOdomLowFreqPublisher;
  ros::Publisher dsoLocalPointCloudPublisher, dsoGlobalPointCloudPublisher, dsoReferencePointCloudPublisher;
  ros::Publisher dsoLocReferencePointCloudPublisher;

  ros::Publisher dsoImagePublisher;
  ros::Subscriber dmvioImageSubscriber;

  ros::ServiceServer dsoGlobalCloudSaver;

  std::function<bool(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)> srv_cbk =
      [this](std_srvs::Empty::Request &req, std_srvs::Empty::Response &res) {
        PointCloudXYZ::Ptr global_cloud_1(new PointCloudXYZ);
        pcl::fromPCLPointCloud2(*global_cloud, *global_cloud_1);
        pcl::io::savePCDFileASCII("stereo_dso_global_cloud.pcd", *global_cloud_1);
        ROS_INFO("Global cloud saved");
        return true;
      };

  // ref cloud
  bool useReferenceCloud;
  std::string referenceCloudPath;
  PointCloudXYZ loc_reference_cloud;

  bool useFiltering;
  pcl::RadiusOutlierRemoval<pcl::PCLPointCloud2> outrem;
  pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;

  pcl::PCLPointCloud2::Ptr global_cloud;

  PointCloudXYZ::Ptr reference_cloud;
  std::deque<PointCloudXYZ::Ptr> margin_cloud_window;
  double lastTimestamp;
  int minNumPointsToSend;
  bool useRANSAC;

  bool isInitialized{false};

  // RANSAC parameters
  double distanceThreshold;
  double probability;
  int maxIterations;

  // RadiusOutlierRemoval parameters
  double activeRadiusSearch, marginRadiusSearch;
  int activeMinNeighborsInRadius, marginMinNeighborsInRadius;

  // StatisticalOutlierRemoval parameters
  int meanK;
  double stddevMulThresh;

  long int pub_idx = 0;

  std::deque<nav_msgs::Odometry> poseBuf;
  std::deque<sensor_msgs::PointCloud2> localPointsBuf, globalPointsBuf;
  std::deque<sensor_msgs::PointCloud2> referencePointsBuf;

  tf2_ros::TransformBroadcaster dsoWcamBr;

  Sophus::SE3d mCamToWorld;
  Sophus::SE3d Toc, Tmir;

  //                Eigen::Matrix4d e_Toc = (Eigen::Matrix4d() << 1, 0, 0, 0,
  //                                         0, 0, 1, 0,
  //                                         0, 1, 0, 0,
  //                                         0, 0, 0, 1)
  //                                                .finished();

  Eigen::Matrix4d e_Toc =
      (Eigen::Matrix4d() << 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();

  Eigen::Matrix4d e_mir_Y =
      (Eigen::Matrix4d() << 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();

  Eigen::Matrix4d e_mir_Z =
      (Eigen::Matrix4d() << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();

  // TODO: use this for now
  // from dm_vio
  //        Eigen::Matrix4d e_Toc = (Eigen::Matrix4d() << 0, 0, 1, 0,
  //                                 -1, 0, 0, 0,
  //                                 0, -1, 0, 0,
  //                                 0, 0, 0, 1)
  //                                        .finished();

  float scaledTH = 1e10;
  float absTH = 1e10;
  float minRelBS = 0;

  // Protects transformDSOToIMU.
  std::mutex mutex;

  // Protects queues
  std::mutex poseMutex, pclMutex;
};

}  // namespace dso
