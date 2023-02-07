/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "FullSystem.h"
#include "OutputWrapper/SampleOutputWrapper.h"
#include "Pangolin/PangolinDSOViewer.h"
#include "Undistort.h"
#include "settings.h"

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include "cv_bridge/cv_bridge.h"

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include "DatasetReader.h"

#include "ROSWrapper/ROSOutputWrapper.h"

std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
bool useSampleOutput = false;
bool preload = false;
int mode = 0;

float playbackSpeed = 0;  // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping).
                          // otherwise, factor on timestamps.

using namespace dso;

void settingsDefault(int preset) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 0 || preset == 1) {
    printf(
        "DEFAULT settings:\n"
        "- %s real-time enforcing\n"
        "- 2000 active points\n"
        "- 5-7 active frames\n"
        "- 1-6 LM iteration each KF\n"
        "- original image resolution\n",
        preset == 0 ? "no " : "1x");

    playbackSpeed = (preset == 0 ? 0 : 1);
    preload = preset == 1;

    setting_desiredImmatureDensity = 1500;  // original 1500. set higher
    setting_desiredPointDensity = 2000;     // original 2000
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    /*林辉灿注释掉
    setting_kfGlobalWeight=0.3;   // original is 1.0. 0.3 is a balance between speed and accuracy. if tracking lost, set
    this para higher setting_maxShiftWeightT= 0.04f * (640 + 128);   // original is 0.04f * (640+480); this para is
    depend on the crop size. setting_maxShiftWeightR= 0.04f * (640 + 128);   // original is 0.0f * (640+480);
    setting_maxShiftWeightRT= 0.02f * (640 + 128);  // original is 0.02f * (640+480);
    */
    /*林辉灿重设参数 KITTI*
    setting_kfGlobalWeight=1.0;   // original is 1.0. 0.3 is a balance between speed and accuracy. if tracking lost, set
    this para higher setting_maxShiftWeightT= 0.04f * (960 + 320);   // original is 0.04f * (640+480); this para is
    depend on the crop size. setting_maxShiftWeightR= 0.04f * (960 + 320);    // original is 0.0f * (640+480);
    setting_maxShiftWeightRT= 0.02f * (960 + 320);  // original is 0.02f * (640+480);
    /*林辉灿重设参数  EuRoC*/
    setting_kfGlobalWeight =
        0.3f;  // original is 1.0. 0.3 is a balance between speed and accuracy. if tracking lost, set this para higher
    setting_maxShiftWeightT =
        0.04f * (640 + 360);  // original is 0.04f * (640+480); this para is depend on the crop size.
    setting_maxShiftWeightR = 0.04f * (640 + 360);   // original is 0.0f * (640+480);
    setting_maxShiftWeightRT = 0.02f * (640 + 360);  // original is 0.02f * (640+480);*/

    setting_logStuff = false;
  } else if (preset == 2 || preset == 3) {
    printf(
        "FAST settings:\n"
        "- %s real-time enforcing\n"
        "- 800 active points\n"
        "- 4-6 active frames\n"
        "- 1-4 LM iteration each KF\n"
        "- 424 x 320 image resolution\n",
        preset == 0 ? "no " : "5x");

    playbackSpeed = (preset == 2 ? 0 : 5);
    preload = preset == 3;
    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;

    setting_logStuff = false;
  }

  printf("==============================================\n");
}

void parseArgument(char *arg) {
  int option;
  char buf[1000];

  if (1 == sscanf(arg, "preset=%d", &option)) {
    settingsDefault(option);
    return;
  }

  if (1 == sscanf(arg, "mode=%d", &option)) {
    mode = option;
    if (option == 0) {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if (option == 1) {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = 0;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = 0;  //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if (option == 2) {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = -1;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = -1;  //-1: fix. >=0: optimize (with prior, if > 0).
      setting_minGradHistAdd = 3;
    }
    return;
  }

  if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
    if (option == 1) {
      useSampleOutput = true;
      printf("USING SAMPLE OUTPUT WRAPPER!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "quiet=%d", &option)) {
    if (option == 1) {
      setting_debugout_runquiet = true;
      printf("QUIET MODE, I'll shut up!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "nolog=%d", &option)) {
    if (option == 1) {
      setting_logStuff = false;
      printf("DISABLE LOGGING!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "nogui=%d", &option)) {
    if (option == 1) {
      disableAllDisplay = true;
      printf("NO GUI!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nomt=%d", &option)) {
    if (option == 1) {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "calib=%s", buf)) {
    calib = buf;
    printf("loading calibration from %s!\n", calib.c_str());
    return;
  }
  if (1 == sscanf(arg, "vignette=%s", buf)) {
    vignetteFile = buf;
    printf("loading vignette from %s!\n", vignetteFile.c_str());
    return;
  }

  if (1 == sscanf(arg, "gamma=%s", buf)) {
    gammaFile = buf;
    printf("loading gammaCalib from %s!\n", gammaFile.c_str());
    return;
  }

  printf("could not parse argument \"%s\"!!\n", arg);
}

std::unique_ptr<FullSystem> fullSystem;
std::unique_ptr<Undistort> undistorter;
IOWrap::PangolinDSOViewer *viewer;
dso::ROSOutputWrapper *rosOutput;
int frameID = 0;
bool stopSystem = false;
int start = 2;

double convertStamp(const ros::Time &time) {
  // We need the timestamp in seconds as double
  return time.sec * 1.0 + time.nsec / 1000000000.0;
}

void run() {
  /*int ii = 0;
  int lastResetIndex = 0;

  while (!stopSystem)
  {
      // Skip the first few frames if the start variable is set.
      if (start > 0 && ii < start)
      {
          ++ii;
          continue;
      }

      auto pair = frameContainer.getImageAndIMUData(frameSkipping.getMaxSkipFrames(frameContainer.getQueueSize()));

      if (!pair.first) continue;

      fullSystem->addActiveFrame(pair.first.get(), ii, &(pair.second), nullptr);

      if (fullSystem->initFailed || setting_fullResetRequested)
      {
          if (ii - lastResetIndex < 250 || setting_fullResetRequested)
          {
              printf("RESETTING!\n");
              std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
              fullSystem.reset();
              for (IOWrap::Output3DWrapper *ow: wraps) ow->reset();

              fullSystem = std::make_unique<FullSystem>(linearizeOperation, imuCalibration, imuSettings);
              if (undistorter->photometricUndist != nullptr)
              {
                  fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
              }
              fullSystem->outputWrapper = wraps;

              setting_fullResetRequested = false;
              lastResetIndex = ii;
          }
      }

      if (viewer != nullptr && viewer->shouldQuit())
      {
          std::cout << "User closed window -> Quit!" << std::endl;
          break;
      }

      if (fullSystem->isLost)
      {
          printf("LOST!!\n");
          break;
      }

      ++ii;

      // Here for all accumulated poses and PCL publish
      // sleep here
      rosOutput.publishOutput();
      //        r.sleep();

      //        ROS_WARN("Time of loop: %f", ticToc->toc());
      delete ticToc;
  }*/
}

void callback(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::ImageConstPtr &img_right) {
  double stamp = convertStamp(img->header.stamp);

  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  assert(cv_ptr->image.type() == CV_8U);
  assert(cv_ptr->image.channels() == 1);

  cv_bridge::CvImagePtr cv_ptr_right = cv_bridge::toCvCopy(img_right, sensor_msgs::image_encodings::MONO8);
  assert(cv_ptr_right->image.type() == CV_8U);
  assert(cv_ptr_right->image.channels() == 1);

  // TODO: check! maybe the first condition can cause issues
  if (fullSystem->initFailed || setting_fullResetRequested) {
    printf("RESETTING!\n");
    std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
    fullSystem.reset();
    for (IOWrap::Output3DWrapper *ow : wraps) ow->reset();

    fullSystem = std::make_unique<FullSystem>();
    fullSystem->linearizeOperation = false;
    fullSystem->outputWrapper = wraps;
    if (undistorter->photometricUndist != 0) fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
    setting_fullResetRequested = false;
  }

  if (fullSystem->isLost) {
    printf("LOST!!\n");
    ros::shutdown();
  }

  rosOutput->publishOutput();

  MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows, (unsigned char *)cv_ptr->image.data);
  MinimalImageB minImg_right((int)cv_ptr_right->image.cols, (int)cv_ptr_right->image.rows,
                             (unsigned char *)cv_ptr_right->image.data);
  ImageAndExposure *undistImg = undistorter->undistort<unsigned char>(&minImg, 1, stamp, 1.0f);
  ImageAndExposure *undistImg_right = undistorter->undistort<unsigned char>(&minImg_right, 1, stamp, 1.0f);

  // TODO: here we should add images
  fullSystem->addActiveFrame(undistImg, undistImg_right, frameID);
  frameID++;
  // printf("frameID: %d\n", frameID);
  delete undistImg;
  delete undistImg_right;

  if (stopSystem) {
    fullSystem->blockUntilMappingIsFinished();

    //        fullSystem->printResult();

    for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper) ow->join();
    // TODO: check if we need to reset the pointer as well

    printf("DELETE FULLSYSTEM!\n");
    undistorter.reset();
    fullSystem.reset();

    ros::shutdown();

    printf("EXIT NOW!\n");
  }
}

/* std::function<void(const sensor_msgs::ImageConstPtr &img,
                    const sensor_msgs::ImageConstPtr &img_right)>
         callback = [](const sensor_msgs::ImageConstPtr &img,
                       const sensor_msgs::ImageConstPtr &img_right)
{
     double stamp = convertStamp(img->header.stamp);

     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
     assert(cv_ptr->image.type() == CV_8U);
     assert(cv_ptr->image.channels() == 1);

     cv_bridge::CvImagePtr cv_ptr_right = cv_bridge::toCvCopy(img_right, sensor_msgs::image_encodings::MONO8);
     assert(cv_ptr_right->image.type() == CV_8U);
     assert(cv_ptr_right->image.channels() == 1);


     if (setting_fullResetRequested)
     {
         std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
         delete fullSystem;
         for (IOWrap::Output3DWrapper *ow: wraps) ow->reset();
         fullSystem = new FullSystem();
         fullSystem->linearizeOperation = false;
         fullSystem->outputWrapper = wraps;
         if (undistorter->photometricUndist != 0)
             fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
         setting_fullResetRequested = false;
     }

     MinimalImageB minImg((int) cv_ptr->image.cols, (int) cv_ptr->image.rows, (unsigned char *) cv_ptr->image.data);
     MinimalImageB minImg_right((int) cv_ptr_right->image.cols, (int) cv_ptr_right->image.rows,
                                (unsigned char *) cv_ptr_right->image.data);
     ImageAndExposure *undistImg = undistorter->undistort<unsigned char>(&minImg, 1, stamp, 1.0f);
     ImageAndExposure *undistImg_right = undistorter->undistort<unsigned char>(&minImg_right, 1, stamp, 1.0f);

     fullSystem->addActiveFrame(undistImg, undistImg_right, frameID);
     frameID++;
     //printf("frameID: %d\n", frameID);
     delete undistImg;
     delete undistImg_right;
 };*/

int main(int argc, char **argv) {
  ros::init(argc, argv, "stereo_dso_ros");
  ros::NodeHandle nh;

  // TODO: m.b.i
  setlocale(LC_ALL, "C");

  for (int i = 1; i < argc; i++) parseArgument(argv[i]);

  undistorter.reset(Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile));

  setGlobalCalib((int)undistorter->getSize()[0], (int)undistorter->getSize()[1], undistorter->getK().cast<float>());

  baseline = undistorter->getBl();

  if (!disableAllDisplay)
    viewer = new IOWrap::PangolinDSOViewer((int)undistorter->getSize()[0], (int)undistorter->getSize()[1]);

  fullSystem = std::make_unique<FullSystem>();
  fullSystem->linearizeOperation = false;

  if (undistorter->photometricUndist != nullptr) fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

  if (viewer) fullSystem->outputWrapper.push_back(viewer);

  if (useSampleOutput) fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

  // TODO: check if we need it
  /*dso::FrameSkippingStrategy frameSkipping(frameSkippingSettings);
  // frameSkipping registers as an outputWrapper to get notified of changes of the system status.
  fullSystem->outputWrapper.push_back(&frameSkipping);*/

  // TODO: check if we need a separate thread
  //    boost::thread runThread = boost::thread(boost::bind(run, viewer.get()));

  message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/cam0/image_raw", 1);   // "/camera/left/image_raw"
  message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/cam1/image_raw", 1);  // "/camera/right/image_raw"
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub, right_sub);
  //    sync.registerCallback(&callback);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  // TODO: make it work with Pangolin (doesn't work if we use ros output wrapper)
  // This will handle publishing to ROS topics.
  rosOutput = new ROSOutputWrapper();
  fullSystem->outputWrapper.push_back(rosOutput);

  ros::spin();
  stopSystem = true;

  return 0;
}
