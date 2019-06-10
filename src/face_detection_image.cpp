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
#define PATH_TO_IR_XML "/models/Retail/object_detection/face/face-detection-retail-0004-fp16.xml"
#define PATH_TO_IR_BIN "/models/Retail/object_detection/face/face-detection-retail-0004-fp16.bin"
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
template <typename T>
void imatU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob,
                 float scale_factor = 1.0, int batch_index = 0)
{
  InferenceEngine::SizeVector blob_size = blob->getTensorDesc().getDims();
  // const size_t width = blob_size[3];
  const int width = blob_size[3];
  // const size_t height = blob_size[2];
  const int height = blob_size[2];
  // const size_t channels = blob_size[1];
  const int channels = blob_size[1];
  T* blob_data = blob->buffer().as<T*>();

  cv::Mat resized_image(orig_image);
  if (width != orig_image.size().width || height != orig_image.size().height)
  {
    cv::resize(orig_image, resized_image, cv::Size(width, height));
    ROS_INFO_STREAM("Image resized" <<width<<" " <<height);
  }
  
  int batchOffset = batch_index * width * height * channels;

  for (int c = 0; c < channels; c++)
  {
    for (int h = 0; h < height; h++)
    {
      for (int w = 0; w < width; w++)
      {
        blob_data[batchOffset + c * width * height + h * width + w] =
            resized_image.at<cv::Vec3b>(h, w)[c] * scale_factor;
            //ROS_INFO_STREAM(resized_image.at<cv::Vec3b>(h, w)[c] * scale_factor);
      }
    }
  }
}

static Blob::Ptr wrapMatToBlob(const cv::Mat& m) {
  std::vector<size_t> reversedShape(&m.size[0], &m.size[0] + m.dims);
  std::reverse(reversedShape.begin(), reversedShape.end());
  return make_shared_blob<float>(Precision::FP32, reversedShape, (float*)m.data);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "face_detection_image");
    ros::NodeHandle n;
    ros::Publisher image_pub = n.advertise<sensor_msgs::Image>("/image_converter/output_video", 1);
    InferencePlugin plugin =PluginDispatcher({""}).getPluginByDevice("MYRIAD");
    printPluginVersion(plugin, std::cout);
    ROS_INFO_STREAM("Plugin device ready");
    CNNNetReader netReader;
    std::string movidius_path = ros::package::getPath("movidius");
    netReader.ReadNetwork(movidius_path + PATH_TO_IR_XML);
    netReader.ReadWeights(movidius_path + PATH_TO_IR_BIN);
    CNNNetwork network = netReader.getNetwork();
    network.setBatchSize(MAX_BATCH_SIZE);
    ROS_INFO_STREAM("Model read");
    auto inputInfo = network.getInputsInfo();
    auto outputInfo = network.getOutputsInfo();
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
  //ImageConverter ic;
  //ros::Rate rate(10);
  //while (ros::ok()){
    cv::Mat image = cv::imread("/home/bot/catkin_ws/src/movidius/face.jpg");

    int width = image.cols;
    int height = image.rows;
    ROS_INFO_STREAM("Image width: " << width);
    ROS_INFO_STREAM("Image height: " << height);
    if (!image.empty()){
        //cv::Mat resized_image(orig_image);
        //cv::resize(300, , cv::Size(width, height));
        imatU8ToBlob<uint8_t >(image, inputBlob);  
        ROS_INFO_STREAM("Matu8ToBlob OK");
        infer_request_curr->Infer();
        ROS_INFO_STREAM("Infer OK");
        /*
        auto const memLocker = outputBlob->cbuffer(); // use const memory locker
        // output_buffer is valid as long as the lifetime of memLocker
        const float *output_buffer = memLocker.as<const float *>();
        /** output_buffer[] - accessing output blob data **/
        const float *detections = infer_request_curr->GetBlob(output_name)->buffer().as<float *>();
        /*for  (int j=0;j<100; j++)
        {
        ROS_INFO_STREAM( std::setprecision(2)<<detections[j]);
        }*/
        const InferenceEngine::SizeVector outputDims = infer_request_curr->GetBlob(output_name)->dims();
        int maxProposalCount = outputDims[1];
        int objectSize = outputDims[0];
        ROS_INFO_STREAM("Dims Ok");
        ROS_INFO_STREAM(outputDims[0]);
        ROS_INFO_STREAM(outputDims[1]);
        ROS_INFO_STREAM(outputDims[2]);
        ROS_INFO_STREAM(outputDims[3]);
        //ROS_INFO_STREAM(objectSize);
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];
            ROS_INFO_STREAM("Image ID:" << image_id);
            if ((image_id < 0) || (image_id >= MAX_BATCH_SIZE)) {  // indicates end of detections
                ROS_INFO_STREAM("Invalid ID:");
                break;
            }
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            ROS_INFO_STREAM("Label: " << r.label);
            r.confidence = detections[i * objectSize + 2];
            ROS_INFO_STREAM("Confidence: "  << r.confidence);
            r.location.x = detections[i * objectSize + 3] * width;
            ROS_INFO_STREAM("X Location: " << r.location.x);
            r.location.y = detections[i * objectSize + 4] * height;
            ROS_INFO_STREAM("Y Location: " << r.location.y);
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            ROS_INFO_STREAM("Width: " << r.location.width);
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;
            ROS_INFO_STREAM("Height: " << r.location.height);
            if (r.confidence > 0.5){
                cv::Rect rect(r.location.x, r.location.y, r.location.width, r.location.height);
                cv::rectangle(image, rect, cv::Scalar(0, 255, 0));
            }
        }
        ROS_INFO_STREAM("For OK");
        cv_bridge::CvImage out_msg;
        //out_msg.header   = in_msg->header; // Same timestamp and tf frame as input image
        out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
        out_msg.image    = image; // Your cv::Mat

        image_pub.publish(out_msg.toImageMsg());

        ROS_INFO_STREAM("Image pub OK");
    //}
    //rate.sleep();
    //ros::spinOnce();

  }
  return 0;
}
