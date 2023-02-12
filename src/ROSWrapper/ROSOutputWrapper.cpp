#include "ROSOutputWrapper.h"
#include <FrameShell.h>
#include <HessianBlocks.h>

using namespace dso;

dso::ROSOutputWrapper::ROSOutputWrapper()
    : nh("stereo_dso"),
      global_cloud(new pcl::PCLPointCloud2),
      //      reference_cloud(new PointCloudXYZ),
      lastTimestamp(0.0) {
  dsoOdomHighFreqPublisher = nh.advertise<nav_msgs::Odometry>("pose_hf", 10);
  dsoOdomLowFreqPublisher = nh.advertise<nav_msgs::Odometry>("pose_lf", 10);
  dsoLocalPointCloudPublisher = nh.advertise<sensor_msgs::PointCloud2>("local_point_cloud", 1);
  dsoLocReferencePointCloudPublisher = nh.advertise<sensor_msgs::PointCloud2>("loc_reference_point_cloud", 1, true);

  dsoGlobalCloudSaver =
      nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>("/dso/save_map", srv_cbk);

  ros::param::get("distanceThreshold", distanceThreshold);
  ros::param::get("probability", probability);
  ros::param::get("maxIterations", maxIterations);
  std::cout << "distanceThreshold: " << distanceThreshold << std::endl;
  std::cout << "probability: " << probability << std::endl;
  std::cout << "maxIterations: " << maxIterations << std::endl;

  ros::param::get("useFiltering", useFiltering);
  std::cout << "useFiltering: " << useFiltering << std::endl;

  ros::param::get("activeRadiusSearch", activeRadiusSearch);
  ros::param::get("activeMinNeighborsInRadius", activeMinNeighborsInRadius);
  std::cout << "activeRadiusSearch: " << activeRadiusSearch << std::endl;
  std::cout << "activeMinNeighborsInRadius: " << activeMinNeighborsInRadius << std::endl;

  ros::param::get("marginRadiusSearch", marginRadiusSearch);
  ros::param::get("marginMinNeighborsInRadius", marginMinNeighborsInRadius);
  std::cout << "marginRadiusSearch: " << marginRadiusSearch << std::endl;
  std::cout << "marginMinNeighborsInRadius: " << marginMinNeighborsInRadius << std::endl;

  ros::param::get("meanK", meanK);
  ros::param::get("stddevMulThresh", stddevMulThresh);
  std::cout << "meanK: " << meanK << std::endl;
  std::cout << "stddevMulThresh: " << stddevMulThresh << std::endl;

  ros::param::get("minNumPointsToSend", minNumPointsToSend);
  ros::param::get("useRANSAC", useRANSAC);

  std::cout << "minNumPointsToSend: " << minNumPointsToSend << std::endl;
  std::cout << "useRANSAC: " << useRANSAC << std::endl;

  ros::param::get("useReferenceCloud", useReferenceCloud);
  std::cout << "useReferenceCloud: " << useReferenceCloud << std::endl;
  ros::param::get("referenceCloudPath", referenceCloudPath);
  std::cout << "referenceCloudPath: " << referenceCloudPath << std::endl;

  pcl::io::loadPCDFile(referenceCloudPath, loc_reference_cloud);
  if (useReferenceCloud) publish_reference_cloud();

  poseBuf.clear();
  localPointsBuf.clear();
  globalPointsBuf.clear();

  margin_cloud_window.clear();

  referencePointsBuf.clear();

  e_Toc.block<3, 3>(0, 0) = Eigen::Quaterniond(e_Toc.block<3, 3>(0, 0)).normalized().toRotationMatrix();
  Toc = Sophus::SE3d(e_Toc);

  Tmir = Sophus::SE3d(e_mir_Y * e_mir_Z);
  Tmir.setRotationMatrix((e_mir_Y * e_mir_Z).block<3, 3>(0, 0));

  Sophus::SO3d R1 = Sophus::SO3d((e_mir_Y * e_mir_Z).block<3, 3>(0, 0));
}

void ROSOutputWrapper::publish_reference_cloud() {
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(loc_reference_cloud, cloud_msg);
  cloud_msg.header.frame_id = "map";
  cloud_msg.header.stamp = ros::Time::now();

  dsoLocReferencePointCloudPublisher.publish(cloud_msg);
  ROS_INFO("Publish reference cloud as latched");
}

void setMsgFromSE3(geometry_msgs::Pose &poseMsg, const Sophus::SE3d &pose) {
  poseMsg.position.x = pose.translation()[0];
  poseMsg.position.y = pose.translation()[1];
  poseMsg.position.z = pose.translation()[2];
  poseMsg.orientation.x = pose.so3().unit_quaternion().x();
  poseMsg.orientation.y = pose.so3().unit_quaternion().y();
  poseMsg.orientation.z = pose.so3().unit_quaternion().z();
  poseMsg.orientation.w = pose.so3().unit_quaternion().w();
}

void setTfFromSE3(geometry_msgs::Transform &tfMsg, const Sophus::SE3d &pose) {
  tfMsg.translation.x = pose.translation()[0];
  tfMsg.translation.y = pose.translation()[1];
  tfMsg.translation.z = pose.translation()[2];
  tfMsg.rotation.x = pose.so3().unit_quaternion().x();
  tfMsg.rotation.y = pose.so3().unit_quaternion().y();
  tfMsg.rotation.z = pose.so3().unit_quaternion().z();
  tfMsg.rotation.w = pose.so3().unit_quaternion().w();
}

Sophus::SE3d transformPointFixedScale(const Sophus::SE3d &pose, Eigen::Vector3d &point_cam) {
  // TODO: we used sim(3) here
  Sophus::SE3d T_w_cam = pose.inverse();
  Sophus::Vector3d t_cam_point(point_cam);
  Sophus::SO3d R_cam_point;
  Sophus::SE3d T_W_point;

  T_W_point = T_w_cam * Sophus::SE3d(R_cam_point, t_cam_point);

  return T_W_point;
}

Sophus::SE3d invTransformPointFixedScale(const Sophus::SE3d &pose, Eigen::Vector3d &point_world) {
  // TODO: we used sim(3) here
  Sophus::SE3d T_w_cam = pose.inverse();
  Sophus::Vector3d t_world_point(point_world);
  Sophus::SO3d R_world_point;
  Sophus::SE3d T_cam_point;

  T_cam_point = T_w_cam.inverse() * Sophus::SE3d(R_world_point, t_world_point);

  return T_cam_point;
}

void ROSOutputWrapper::publishOutput() {
  if (poseBuf.empty() || localPointsBuf.empty()) {
    return;
  }

  std::unique_ptr<nav_msgs::Odometry> dso_pose = nullptr;
  poseMutex.lock();
  while (poseBuf.size() > 1) poseBuf.pop_front();
  dso_pose = std::make_unique<nav_msgs::Odometry>(poseBuf.front());
  dsoOdomLowFreqPublisher.publish(*dso_pose);
  poseBuf.clear();
  poseMutex.unlock();

  if (dso_pose == nullptr) {
    ROS_WARN("Locked the pose. Nothing to publish");
    return;
  }

  sensor_msgs::PointCloud2 dso_local_cloud;
  pclMutex.lock();

  auto publish_point_cloud = [&](std::deque<sensor_msgs::PointCloud2> &points_buf, sensor_msgs::PointCloud2 &msg,
                                 const ros::Publisher &publisher) {
    while (points_buf.size() > 1) points_buf.pop_front();
    msg = points_buf.front();
    msg.header.stamp = dso_pose->header.stamp;
    publisher.publish(msg);
    points_buf.clear();
  };

  publish_point_cloud(localPointsBuf, dso_local_cloud, dsoLocalPointCloudPublisher);

  pclMutex.unlock();
}

void ROSOutputWrapper::publishInitSignal() {
  if (!isInitialized) {
    ROS_INFO("System is initialized. Start publishing poses");
    isInitialized = true;
  }
}

void ROSOutputWrapper::publishCamPose(dso::FrameShell *frame, dso::CalibHessian *HCalib) {
  ros::Time ros_ts;
  ros_ts.fromSec(frame->timestamp);
  lastTimestamp = frame->timestamp;

  auto &camToWorld = frame->camToWorld;

  {
    std::unique_lock<std::mutex> lk(mutex);
    if (!isInitialized) return;

    nav_msgs::Odometry metricOdomMsg;
    metricOdomMsg.header.stamp = ros_ts;
    metricOdomMsg.header.frame_id = "odom";
    metricOdomMsg.child_frame_id = "zed2_camera_frame";

    mCamToWorld = camToWorld;
    setMsgFromSE3(metricOdomMsg.pose.pose, Tmir * Toc * camToWorld * Toc.inverse() * Tmir.inverse());

    geometry_msgs::TransformStamped tf_cam;
    tf_cam.header = metricOdomMsg.header;
    tf_cam.header.frame_id = "odom";
    tf_cam.child_frame_id = "zed2_camera_frame";
    setTfFromSE3(tf_cam.transform, Tmir * Toc * camToWorld * Toc.inverse() * Tmir.inverse());

    {
      std::unique_lock<std::mutex> mtx(poseMutex);
      poseBuf.push_back(metricOdomMsg);
    }
    dsoOdomHighFreqPublisher.publish(metricOdomMsg);
    dsoWcamBr.sendTransform(tf_cam);
  }
}

void ROSOutputWrapper::pushLiveFrame(dso::FrameHessian *image) {
  //    cv_bridge::CvImage cvImage;
  //    cvImage.encoding = "mono8";
  //
  //    int cols = 848;
  //    int rows = 800;
  //    unsigned char *img_data = new unsigned char[cols * rows];
  //    for (int i = 0; i < cols * rows; i++)
  //        img_data[i] = image->dI[i][0] * 0.8 > 255.0f ? 255.0 : image->dI[i][0] * 0.8;
  //    cv::Mat cv_mat_image(rows, cols, CV_8UC1, &img_data[0]);
  //    cvImage.image = cv_mat_image;
  //
  //    sensor_msgs::Image imageMsg;
  //    cvImage.toImageMsg(imageMsg);
  //    delete[] img_data;
  //
  //    ros::Time ros_ts;
  //    ros_ts.fromSec(image->shell->timestamp);
  //    imageMsg.header.stamp = ros_ts;
  //    dsoImagePublisher.publish(imageMsg);
}

void ROSOutputWrapper::publishKeyframes(std::vector<dso::FrameHessian *> &frames, bool final,
                                        dso::CalibHessian *HCalib) {
  if (!isInitialized) return;

  float fx = HCalib->fxl();
  float fy = HCalib->fyl();
  float cx = HCalib->cxl();
  float cy = HCalib->cyl();

  float fxi = 1 / fx;
  float fyi = 1 / fy;
  float cxi = -cx / fx;
  float cyi = -cy / fy;

  sensor_msgs::PointCloud2 msg_local_cloud;
  PointCloudXYZ::Ptr temp_active_local_cloud_world(new PointCloudXYZ);
  PointCloudXYZ::Ptr temp_margin_local_cloud_world(new PointCloudXYZ);

  pcl::PCLPointCloud2::Ptr active_local_cloud_world(new pcl::PCLPointCloud2);
  pcl::PCLPointCloud2::Ptr margin_local_cloud_world(new pcl::PCLPointCloud2);

  size_t npointsHessians = 0;
  size_t npointsHessiansMarginalized = 0;
  double timestamp = 0.0;

  {
    std::unique_lock<std::mutex> lk(mutex);
    for (dso::FrameHessian *fh : frames) {
      npointsHessians += fh->pointHessians.size();
      npointsHessiansMarginalized += fh->pointHessiansMarginalized.size();

      auto fill_point_cloud = [&](const std::vector<dso::PointHessian *> &points, PointCloudXYZ::Ptr cloud) {
        for (auto &ph : points) {
          Eigen::Vector3d pos_cam, pos_world, pos_metric_cam;

          // [sx, sy, s]
          float idpeth = ph->idepth_scaled;
          float idepth_hessian = ph->idepth_hessian;
          float relObsBaseline = ph->maxRelBaseline;

          if (idpeth < 0) continue;

          float depth = (1.0f / idpeth);
          float depth4 = depth * depth;
          depth4 *= depth4;
          float var = (1.0f / (idepth_hessian + 0.01f));

          if (var * depth4 > scaledTH) continue;

          if (var > absTH) continue;

          if (relObsBaseline < minRelBS) continue;

          // TODO: we used fixed scaling here
          pos_cam[0] = (ph->u * fxi + cxi) * depth;
          pos_cam[1] = (ph->v * fyi + cyi) * depth;
          pos_cam[2] = depth * (1 + 2 * fxi * (rand() / static_cast<float>(RAND_MAX) - 0.5f));

          auto &camToWorld = fh->shell->camToWorld;
          Sophus::SE3d fixedScalePointToWorld(transformPointFixedScale(camToWorld.inverse(), pos_cam));

          Eigen::Matrix<double, 4, 4> e_fixedScalePointToWorld = fixedScalePointToWorld.matrix();

          for (int i = 0; i < 3; i++) pos_world[i] = e_fixedScalePointToWorld(i, 3);

          // TODO: make for cam points

          Sophus::SE3d fixedScalePointToCam(invTransformPointFixedScale(mCamToWorld.inverse(), pos_world));

          // If you don't want to use the optical transform, then remove Toc from here
          Eigen::Matrix<double, 4, 4> e_fixedScalePointToCam = Tmir.matrix() * Toc.matrix() *
                                                               fixedScalePointToCam.matrix() * Toc.inverse().matrix() *
                                                               Tmir.inverse().matrix();

          for (int i = 0; i < 3; i++) {
            pos_metric_cam[i] = e_fixedScalePointToCam(i, 3);
          }

          pcl::PointXYZ point_world, point_metric_cam;
          point_world.x = pos_world(0);
          point_world.y = pos_world(1);
          point_world.z = pos_world(2);

          point_metric_cam.x = pos_metric_cam(0);
          point_metric_cam.y = pos_metric_cam(1);
          point_metric_cam.z = pos_metric_cam(2);

          cloud->push_back(point_metric_cam);
        }
      };

      fill_point_cloud(fh->pointHessians, temp_active_local_cloud_world);
      fill_point_cloud(fh->pointHessiansMarginalized, temp_margin_local_cloud_world);

      timestamp = fh->shell->timestamp;
    }
  }

  if (temp_active_local_cloud_world->size() < 1 or temp_margin_local_cloud_world->size() < 1) return;

  pcl::toPCLPointCloud2(*temp_active_local_cloud_world, *active_local_cloud_world);
  pcl::toPCLPointCloud2(*temp_margin_local_cloud_world, *margin_local_cloud_world);

  pcl::PCLPointCloud2::Ptr filtered_active_local_cloud_world(new pcl::PCLPointCloud2);
  pcl::PCLPointCloud2::Ptr filtered_margin_local_cloud_world(new pcl::PCLPointCloud2);

  auto filter_point_cloud = [&](pcl::PCLPointCloud2::Ptr raw_cloud, const double &radius_search,
                                const double &min_neighbors_in_radius, pcl::PCLPointCloud2::Ptr filtered_cloud) {
    outrem.setInputCloud(raw_cloud);
    outrem.setRadiusSearch(radius_search);
    outrem.setMinNeighborsInRadius(min_neighbors_in_radius);
    outrem.setKeepOrganized(true);
    outrem.filter(*filtered_cloud);
  };

  if (useFiltering) {
    filter_point_cloud(active_local_cloud_world, activeRadiusSearch, activeMinNeighborsInRadius,
                       filtered_active_local_cloud_world);
    filter_point_cloud(margin_local_cloud_world, marginRadiusSearch, marginMinNeighborsInRadius,
                       filtered_margin_local_cloud_world);
  }

  ros::Time ros_ts;
  ros_ts.fromSec(timestamp);

  if (useFiltering) {
    *global_cloud += *filtered_margin_local_cloud_world;
    pcl_conversions::moveFromPCL(*filtered_margin_local_cloud_world, msg_local_cloud);
  } else {
    *global_cloud += *margin_local_cloud_world;
    pcl_conversions::moveFromPCL(*margin_local_cloud_world, msg_local_cloud);
  }
  msg_local_cloud.header.stamp = ros_ts;
  msg_local_cloud.header.frame_id = "zed2_camera_frame";

  {
    std::unique_lock<std::mutex> mtx(pclMutex);
    localPointsBuf.push_back(msg_local_cloud);
  }
}
