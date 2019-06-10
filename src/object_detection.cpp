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


#include <iostream>
#include <cstdio>
#include <ctime>
//#define PATH_TO_IR_BIN "/models/ssd512_fp16.bin"
//#define PATH_TO_IR_XML "/models/ssd512_fp16.xml"
//#define PATH_TO_IR_XML "/models/mobilenet_ssd_fp16.xml"
//#define PATH_TO_IR_BIN "/models/mobilenet_ssd_fp16.bin"
//#define PATH_TO_IR_XML "/models/Retail/object_detection/face_pedestrian/face-person-detection-retail-0002-fp16.xml"
//#define PATH_TO_IR_BIN "/models/Retail/object_detection/face_pedestrian/face-person-detection-retail-0002-fp16.bin"
//#define PATH_TO_IR_XML "/models/Retail/object_detection/pedestrian/person-detection-retail-0002.xml"
//#define PATH_TO_IR_BIN "/models/Retail/object_detection/pedestrian/person-detection-retail-0002.bin"

#define MAX_BATCH_SIZE 1

static const std::string OPENCV_WINDOW = "Image window";
using namespace InferenceEngine;

struct Result {
        int label;
        float confidence;
        cv::Rect location;
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
    image_sub_ = it_.subscribe("/color/image_raw", 10,
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 10);

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
  std::clock_t start;
   double duration;
    ros::init(argc, argv, "image_converter");
    ros::NodeHandle nh;
    std::string model_path_bin, model_path_xml;
    if(!nh.getParam("model_path_xml", model_path_xml)){
        model_path_xml = "/models/ssd512_fp16.xml";
    }
    if(!nh.getParam("model_path_bin", model_path_bin)){
        model_path_bin = "/models/ssd512_fp16.bin";
    }
    ROS_INFO_STREAM(model_path_xml);
    ROS_INFO_STREAM(model_path_bin);
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
    /*
    if (inputInfo.size()!=1){
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
    for (auto &item : outputInfo) {
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
  cv::Point point;
  std::map<int, std::string> mymap = {
                { 1, "Aeroplane" },
                { 2, "Bicycle" },
                { 3, "Bird" },
                { 4, "Boat" },
                { 5, "Bottle" },
                { 6, "Bus" },
                { 7, "Car" },
                { 8, "Cat" },
                { 9, "Chair" },
                { 10, "Cow" },
                { 11, "Dinningtable" },
                { 12, "Dog" },
                { 13, "Horse" },
                { 14, "Motorbike" },
                { 15, "Person" },
                { 16, "Pottedplant" },
                { 17, "Sheep" },
                { 18, "Sofa" },
                { 19, "Train" },
                { 20, "TV Monitor"  } };
  cv::Mat image;
  do{
    ROS_INFO_STREAM("Getting first image");
    ros::spinOnce();
    image = ic.getImage();
  }while (image.empty());
  int width = image.cols;
  int height = image.rows;
  matU8ToBlob<uint8_t >(image, inputBlob);
  infer_request_curr->StartAsync();
  cv::Mat prev_image = image;
  while (ros::ok()){
    start = std::clock();
    do{
      ROS_INFO_STREAM("Getting next image");
      ros::spinOnce();
      image = ic.getImage();
    }while (image.empty());
    width = image.cols;
    height = image.rows;
    infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY);

    const float *detections = infer_request_curr->GetBlob(output_name)->buffer().as<float *>();
    const InferenceEngine::SizeVector outputDims = infer_request_curr->GetBlob(output_name)->dims();
    int maxProposalCount = outputDims[1];
    int objectSize = outputDims[0];

    matU8ToBlob<uint8_t >(image, inputBlob);
    infer_request_curr->StartAsync();

    for (int i = 0; i < maxProposalCount; i++) {
        float image_id = detections[i * objectSize + 0];
        if ((image_id < 0) || (image_id >= MAX_BATCH_SIZE)) {  // indicates end of detections
            ROS_INFO_STREAM("Invalid ID:");
            break;
        }
        Result r;
        r.label = static_cast<int>(detections[i * objectSize + 1]);
        r.confidence = detections[i * objectSize + 2];
        r.location.x = detections[i * objectSize + 3] * width;
        r.location.y = detections[i * objectSize + 4] * height;
        r.location.width = detections[i * objectSize + 5] * width - r.location.x;
        r.location.height = detections[i * objectSize + 6] * height - r.location.y;
        if (r.confidence > 0.75){
            cv::Rect rect(r.location.x, r.location.y, r.location.width, r.location.height);
            cv::rectangle(prev_image, rect, cv::Scalar(0, 255, 0));
      /*      ROS_INFO_STREAM("Image ID:" << image_id);
            ROS_INFO_STREAM("Label: " << r.label);
            ROS_INFO_STREAM("Confidence: "  << r.confidence);
            ROS_INFO_STREAM("X Location: " << r.location.x);
            ROS_INFO_STREAM("Y Location: " << r.location.y);
            ROS_INFO_STREAM("Width: " << r.location.width);
            ROS_INFO_STREAM("Height: " << r.location.height);
            */
            std::ostringstream str;
            str << "LABEL:" << mymap.at(r.label) ;
            point.x=r.location.x;
            point.y=r.location.y;

            cv::putText(prev_image, str.str(), point, CV_FONT_HERSHEY_PLAIN,1, CV_RGB(0, r.label*100, 0),2);

        }
    }

    ic.publishImage(prev_image );
    prev_image = image;
        //ROS_INFO_STREAM("For OK");

        //ROS_INFO_STREAM("Image pub OK");
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"printf: "<< duration <<"start: "<< start/(double)CLOCKS_PER_SEC << "now: "<<std::clock()/(double)CLOCKS_PER_SEC<<'\n';


  }
  return 0;
}
