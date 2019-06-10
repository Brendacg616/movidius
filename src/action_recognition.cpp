#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <opencv2/opencv.hpp>

#include <action_detector.hpp>

//#define PATH_TO_IR_XML "/models/mobilenet_ssd_fp16.xml"
//#define PATH_TO_IR_BIN "/models/mobilenet_ssd_fp16.bin"
//#define PATH_TO_IR_XML "/models/Retail/object_detection/face_pedestrian/face-person-detection-retail-0002-fp16.xml"
 //#define PATH_TO_IR_BIN "/models/Retail/object_detection/face_pedestrian/face-person-detection-retail-0002-fp16.bin"
//#define PATH_TO_IR_XML "/models/person-detection-retail-0002-fp16.xml"
//#define PATH_TO_IR_BIN "/models/person-detection-retail-0002-fp16.bin"
#define FEATURE_WIDTH 43
#define FEATURE_HEIGHT 25
#define CLASSES 4
#define MAX_BATCH_SIZE 1

static const std::string OPENCV_WINDOW = "Image window";
using namespace InferenceEngine;

struct Result {
        float b;
        float c;
        float h;
        float w;

};

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  cv::Mat image;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/color/image_raw", 1,
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    image = cv_ptr->image;
  }
  cv::Mat getImage(){
      return image;
  }

  void publishImage(cv::Mat image ){
    cv_bridge::CvImage out_msg;
    //out_msg.header   = in_msg->header; // Same timestamp and tf frame as input image
    out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
    out_msg.image    = image; // Your cv::Mat

    image_pub_.publish(out_msg.toImageMsg());
  }
};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ros::NodeHandle nh;
  std::string model_path_bin, model_path_xml;
  ActionDetectorConfig config_;
  if(!nh.getParam("model_path_xml", model_path_xml)){
      model_path_xml = "/models/face-detection-retail-0004-fp16.xml";
  }
  if(!nh.getParam("model_path_bin", model_path_bin)){
      model_path_bin = "/models/face-detection-retail-0004-fp16.bin";
  }
  ROS_INFO_STREAM(model_path_xml);
  ROS_INFO_STREAM(model_path_bin);

ActionDetectorConfig config;
ActionDetection detector(config);

    InferencePlugin plugin =PluginDispatcher({""}).getPluginByDevice("MYRIAD");
    printPluginVersion(plugin, std::cout);
    ROS_INFO_STREAM("Plugin device ready");
    CNNNetReader netReader;
    std::string movidius_path = ros::package::getPath("movidius");
    netReader.ReadNetwork(movidius_path + model_path_xml);
    netReader.ReadWeights(movidius_path + model_path_bin);
    CNNNetwork network = netReader.getNetwork();
    network.setBatchSize(MAX_BATCH_SIZE);
    ROS_INFO_STREAM("Model read");
    auto inputInfo = network.getInputsInfo();
    auto outputInfo = network.getOutputsInfo();
    for (auto &item : inputInfo) {
        auto input_name = item.first;
        ROS_INFO_STREAM(input_name);
        auto input_data = item.second;
        /*InferenceEngine::SizeVector inputDims = infer_request_curr->GetBlob(input_name)->dims();
        //put_data->setPrecision(Precision::FP32);
        ROS_INFO_STREAM(inputDims[0]);
        ROS_INFO_STREAM(inputDims[1]);
        ROS_INFO_STREAM(inputDims[2]);
        ROS_INFO_STREAM(intputDims[3]);*/
    }
/*    if (inputInfo.size()!=1){
      throw std::logic_error("Error");
    }
*/
    ROS_INFO_STREAM(inputInfo.size());
    auto input_name = inputInfo.begin()->first;
    ROS_INFO_STREAM(input_name);
    auto output_name = outputInfo.begin()->first;
    ROS_INFO_STREAM(output_name);
    InputInfo::Ptr & input_params= inputInfo.begin()->second;
    input_params->setPrecision(Precision::U8);
    ROS_INFO_STREAM("Set Precision");
    input_params->getInputData()->setLayout(Layout::NCHW);
    ROS_INFO_STREAM("Set Layout");

    //auto outputInfo  = network.getOutputsInfo();
    std::vector<std::string> output_names;
    for (auto &item : outputInfo) {
      ROS_INFO_STREAM(item.first);
      output_names.push_back(item.first);
      auto output_data = item.second;
      output_data->setPrecision(Precision::FP32);
      //output_data->setLayout(Layout::NC);
    }
    auto executable_network = plugin.LoadNetwork(network, {});
    ROS_INFO_STREAM("Network Loaded");
    auto infer_request_curr = executable_network.CreateInferRequestPtr();
    ROS_INFO_STREAM("Creating inference request");
    auto inputBlob = infer_request_curr->GetBlob(input_name);
    auto outputBlob = infer_request_curr->GetBlob(output_name);
    ROS_INFO_STREAM("Input Blobs");
    ImageConverter ic;
    ros::Rate rate(10);
    cv::Mat image;
    int width, height, i;
    const InferenceEngine::SizeVector outputDims;
    int maxProposalCount;
    int objectSize;
    int objSize2;
    float image_id;
    Result r;
    cv::Point point;
    while (ros::ok()){
    image = ic.getImage();
    width = image.cols;
    height = image.rows;
    if (!image.empty()){
        matU8ToBlob<uint8_t >(image, inputBlob);
        ROS_INFO_STREAM("Matu8ToBlob OK");
        infer_request_curr->Infer();
        ROS_INFO_STREAM("Infer OK");
        /*
        auto const memLocker = outputBlob->cbuffer(); // use const memory locker
        // output_buffer is valid as long as the lifetime of memLocker
        const float *output_buffer = memLocker.as<const float *>();
        /** output_buffer[] - accessing output blob data **/
/*[ INFO] [1556297574.610631705]: mbox/priorbox
[ INFO] [1556297574.610680232]: mbox_loc1/out/conv/flat
[ INFO] [1556297574.610729508]: mbox_main_conf/out/conv/flat/softmax/flat
[ INFO] [1556297574.610778247]: out/anchor1
[ INFO] [1556297574.610822648]: out/anchor2
[ INFO] [1556297574.610868212]: out/anchor3
[ INFO] [1556297574.610914639]: out/anchor4
*/
        detector.fetchResults(infer_request_curr);
        //ROS_INFO_STREAM(detector.results.size());
        if (!detector.results.empty())
        {
        //ROS_INFO_STREAM(detector.results[0].label);
        //ROS_INFO_STREAM(detector.results[0].detection_conf);

        //void DrawObject(cv::Rect rect, const std::string& label_to_draw,
          //                const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg)
          std::map<int, std::string> label_map = {
                { 0, "Sitting" },
                { 1, "Standing" },
                { 2, "Raising Hand" },
                { 3, "Nothing" }};
          for (int i =0 ;i <detector.results.size() ; i++)
              {
                ROS_INFO_STREAM(detector.results[i].label);
                ROS_INFO_STREAM(detector.results[i].detection_conf);
                DetectedAction temp=detector.results[i];

                ROS_INFO_STREAM("x: " << temp.rect.x);
                ROS_INFO_STREAM("y: " << temp.rect.y);
                ROS_INFO_STREAM("width: " << temp.rect.width);
                ROS_INFO_STREAM("height: " << temp.rect.height);
                  if (width/680 != 1 || height/400 != 1)
                  {
                      temp.rect.x = cvRound(temp.rect.x * width/680 );
                      temp.rect.y = cvRound(temp.rect.y * height/400 );
                  //    ROS_INFO_STREAM("x: " << temp.rect.x);
                  //    ROS_INFO_STREAM("y: " << temp.rect.y);
                  //    ROS_INFO_STREAM(detector.results[i].detection_conf);
                      temp.rect.height = cvRound(temp.rect.height * height/400 );
                      temp.rect.width = cvRound(temp.rect.width * width/680);
                      putText(image, label_map.at(temp.label), cv::Point(temp.rect.x, temp.rect.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0), 1, cv::LINE_AA);

                  }
                  cv::rectangle(image, temp.rect, cv::Scalar(255,0,0));
                }
                  //if (plot_bg && !label_to_draw.empty()) {
                      int baseLine = 0;
                      //const cv::Size label_size =
                          //cv::getTextSize(detector.results[i].label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
                    //  cv::rectangle(image, cv::Point(rect.x, rect.y - label_size.height),
                    //cv::Point(rect.x + label_size.width, rect.y + baseLine),
                      //              cv::Scalar(0,255,0), cv::FILLED);
                //  }
                  //if (!label_to_draw.empty()) {
                      //
              }



      //}
        //objectSize = outputDims[2];//3





//        ROS_INFO_STREAM("For OK");
        ic.publishImage(image );
//        ROS_INFO_STREAM("Image pub OK");
    }
    rate.sleep();
    ros::spinOnce();
    //ros::spin();

  }
  return 0;
}
