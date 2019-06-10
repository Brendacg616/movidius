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
    image_sub_ = it_.subscribe("/color/image_raw", 1,
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    ROS_INFO_STREAM("suscriptor imagen");
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
    std::string model_path_bin_face, model_path_xml_face;

    if(!nh.getParam("model_path_xml", model_path_xml)){
        model_path_xml = "/models/ssd512_fp16.xml";
    }
    if(!nh.getParam("model_path_bin", model_path_bin)){
        model_path_bin = "/models/ssd512_fp16.bin";
    }

    if(!nh.getParam("model_path_xml_face", model_path_xml_face)){
        model_path_xml_face = "/models/face-detection-retail-0004-fp16.xml";
    }
    if(!nh.getParam("model_path_bin_face", model_path_bin_face)){
        model_path_bin_face = "/models/face-detection-retail-0004-fp16.bin";
    }


    ROS_INFO_STREAM(model_path_xml);
    ROS_INFO_STREAM(model_path_bin);

    ROS_INFO_STREAM(model_path_xml_face);
    ROS_INFO_STREAM(model_path_bin_face);

    InferencePlugin plugin =PluginDispatcher({""}).getPluginByDevice("MYRIAD");
    printPluginVersion(plugin, std::cout);
    ROS_INFO_STREAM("Plugin device ready");
    CNNNetReader netReader;
    CNNNetReader netReader2;
    std::string movidius_path = ros::package::getPath("movidius");
    netReader.ReadNetwork(movidius_path + model_path_xml);
    netReader.ReadWeights(movidius_path + model_path_bin);

    netReader2.ReadNetwork(movidius_path + model_path_xml_face);
    netReader2.ReadWeights(movidius_path + model_path_bin_face);

    CNNNetwork network = netReader.getNetwork();
    CNNNetwork network2 = netReader2.getNetwork();
    network2.setBatchSize(MAX_BATCH_SIZE);
    network.setBatchSize(MAX_BATCH_SIZE);
    ROS_INFO_STREAM("Model read");
    auto inputInfo = network.getInputsInfo();
    auto outputInfo = network.getOutputsInfo();

    auto inputInfo2 = network2.getInputsInfo();
    auto outputInfo2 = network2.getOutputsInfo();
    /*
    if (inputInfo.size()!=1){
      throw std::logic_error("Error");
    }
*/
    ROS_INFO_STREAM(inputInfo.size());
    auto input_name = inputInfo.begin()->first;
    auto input_name2 = inputInfo2.begin()->first;

    ROS_INFO_STREAM(input_name);
    auto output_name = outputInfo.begin()->first;
    ROS_INFO_STREAM(output_name);
    InputInfo::Ptr & input_params= inputInfo.begin()->second;
    input_params->setPrecision(Precision::U8);
    ROS_INFO_STREAM("Set Precision");
    input_params->getInputData()->setLayout(Layout::NCHW);
    ROS_INFO_STREAM("Set Layout");

    auto output_name2 = outputInfo2.begin()->first;
    ROS_INFO_STREAM(output_name2);
    //auto outputInfo  = network.getOutputsInfo();
    for (auto &item : outputInfo) {
      auto output_data = item.second;
      output_data->setPrecision(Precision::FP32);
      //output_data->setLayout(Layout::NC);
    }



    for (auto &item : inputInfo2) {
        auto input_data = item.second;
        input_data->setPrecision(Precision::U8);
        input_data->setLayout(Layout::NCHW);
        //input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    }
    for (auto &item : outputInfo2) {
      auto output_data2 = item.second;
      output_data2->setPrecision(Precision::FP32);
      //output_data->setLayout(Layout::NC);
    }

    auto executable_network = plugin.LoadNetwork(network, {});
    auto executable_network2 = plugin.LoadNetwork(network2, {});
    ROS_INFO_STREAM("Network Loaded");
    auto infer_request_curr = executable_network.CreateInferRequestPtr();
    auto infer_request2 = executable_network2.CreateInferRequestPtr();
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
  //infer_request_curr->StartAsync();
  infer_request_curr->Infer();
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
    //infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY);

    const float *detections = infer_request_curr->GetBlob(output_name)->buffer().as<float *>();
    const InferenceEngine::SizeVector outputDims = infer_request_curr->GetBlob(output_name)->dims();
    int maxProposalCount = outputDims[1];
    int objectSize = outputDims[0];

    //infer_request_curr->StartAsync();

    //vector rois<InferenceEngine::ROI>;//rois vector

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
        /*size_t 	id

size_t 	posX

size_t 	posY

size_t 	sizeX

size_t 	sizeY*/
        if (r.confidence > 0.5){
            //cv::Rect rect(r.location.x, r.location.y, r.location.width, r.location.height);
            //cv::rectangle(prev_image, rect, cv::Scalar(0, 255, 0));
            if(r.label==15)
            {
            InferenceEngine::ROI temp;
            temp.id=i;
            temp.posX=r.location.x/width;
            temp.posY=r.location.y/height;
            temp.sizeX=r.location.width/width;
            temp.sizeY=r.location.height/height;
            //rois.push_back(temp);
            //auto roiBlob = InferenceEngine::make_shared_blob(inputBlob, temp);
            ROS_INFO_STREAM("SETBLOB FACE");
            auto inputBlob2 = infer_request2->GetBlob(input_name2);
            matU8ToBlob<uint8_t >(prev_image, inputBlob2);
            //infer_request2->GetBlob(input_name);
            ROS_INFO_STREAM("INFER FACE");
            infer_request2->Infer();
            //infer_request2->StartAsync();
            //ROS_INFO_STREAM("INFER FACE");
            //infer_request2->Wait(IInferRequest::WaitMode::RESULT_READY);
            ROS_INFO_STREAM("DETECTIONS FACE");
            const float *detections2 = infer_request2->GetBlob(output_name2)->buffer().as<float *>();
            for (int j = 0; j < maxProposalCount; j++)
             {

            Result f;
            f.label = static_cast<int>(detections2[j * objectSize + 1]);
            f.confidence = detections2[j * objectSize + 2];
            f.location.x = detections2[j * objectSize + 3] * width;
            f.location.y = detections2[j * objectSize + 4] * height;
            f.location.width = detections2[j * objectSize + 5] * width - f.location.x;
            f.location.height = detections2[j * objectSize + 6] * height - f.location.y;

              if(f.confidence>0.9)
              {
                //cv::Rect rect2(f.location.x, f.location.y, f.location.width, f.location.height);
                //cv::rectangle(prev_image, rect2, cv::Scalar(0, 0, 255));
                ROS_INFO_STREAM("Cara encontrada ");
                ROS_INFO_STREAM("confidence: " << f.confidence);
                ROS_INFO_STREAM("X Location: " << f.location.x);
                ROS_INFO_STREAM("Y Location: " << f.location.y);
              }
             }
            }
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

            //cv::putText(prev_image, str.str(), point, CV_FONT_HERSHEY_PLAIN,1, CV_RGB(0, r.label*100, 0),2);

        }
    }

    ic.publishImage(prev_image );
    prev_image = image;

    matU8ToBlob<uint8_t >(image, inputBlob);
    infer_request_curr->Infer();

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"printf: "<< duration <<'\n';


  }
  return 0;
}
