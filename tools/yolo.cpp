#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <opencv2/opencv.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include <stdio.h>
#include <malloc.h>
#include <fstream>
#include <boost/progress.hpp>


#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


//#include "caffe/util/math_functions.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::Layer;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
Dtype lap(Dtype x1_min,Dtype x1_max,Dtype x2_min,Dtype x2_max){
    if(x1_min < x2_min){
        if(x1_max < x2_min){
            return 0;
        }else{
            if(x1_max > x2_min){
                if(x1_max < x2_max){
                    return x1_max - x2_min;
                }else{
                    return x2_max - x2_min;
                }
            }else{
                return 0;
            }
        }
    }else{
        if(x1_min < x2_max){
            if(x1_max < x2_max)
                return x1_max-x1_min;
            else
                return x2_max-x1_min;
        }else{
            return 0;
        }
    }
}

template int lap(int x1_min, int x1_max, int x2_min, int x2_max);
template float lap(float x1_min, float x1_max, float x2_min, float x2_max);

void loadweights(boost::shared_ptr<Net<float> >& net,char* argv);
void loaddata(boost::shared_ptr<Net<float> >& net, std::string image_path);
void getbox(std::vector<float> result,float* pro_obj,int* idx_class,std::vector<std::vector<int> >& bboxs,float thresh,cv::Mat image);
//int lap(int x1_min,int x1_max,int x2_min,int x2_max);
int main(int argc, char** argv){
  //boost::progress_timer t;
  char *labelname[] = {"face"};
  Caffe::set_mode(Caffe::GPU);
  boost::shared_ptr<Net<float> > net(new Net<float>(argv[1], caffe::TEST));
  //loadweights(net,argv[2]);//这行代码是还没caffemodel的时候使用的。
  net->CopyTrainedLayersFromBinaryProto(argv[2]);//有caffemodel的时候就可以使用这行代码，跟上面一行互补使用。
  loaddata(net,std::string(argv[3]));
  //std::cout<<"load weights and data 's time = "<< t.elapsed() <<std::endl;
  net->Forward();
  //std::cout<<"to process finish time = "<< t.elapsed() <<std::endl;
  Blob<float>* output_layer = net->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  std::vector<float> result(begin, end);
  //接下来就是生成框。
  std::vector<std::vector<int> > bboxs;
  float pro_obj[121][2];
  int idx_class[121];
  cv::Mat image = cv::imread(argv[3]);
  float overlap;
  float overlap_thresh = 0.4;
  //cv::resize(image,image,cv::Size(448,448));
  getbox(result, &pro_obj[0][0],idx_class,bboxs,0.2,image);
  std::vector<bool> mark(bboxs.size(),true);
  for(int i = 0; i < bboxs.size(); ++i){
      for(int j = i+1; j < bboxs.size(); ++j){
          int overlap_x = lap(bboxs[i][0],bboxs[i][2],bboxs[j][0],bboxs[j][2]);
          int overlap_y = lap(bboxs[i][1],bboxs[i][3],bboxs[j][1],bboxs[j][3]);
          overlap = (overlap_x*overlap_y)*1.0/((bboxs[i][0]-bboxs[i][2])*(bboxs[i][1]-bboxs[i][3])+(bboxs[j][0]-bboxs[j][2])*(bboxs[j][1]-bboxs[j][3])-(overlap_x*overlap_y));
          if(overlap > overlap_thresh){
              if(bboxs[i][4] > bboxs[j][4]){
                  mark[j] = false;
              }else{
                  mark[i] = false;
              }
          }
      }
  }
  for(int i = 0; i < bboxs.size();++i){
      if(mark[i]){
          cv::Point point1(bboxs[i][1],bboxs[i][2]);
          cv::Point point2(bboxs[i][3],bboxs[i][4]);
          cv::rectangle(image, cv::Rect(point1,point2),cv::Scalar(0,bboxs[i][0]/20.0*225,255),bboxs[i][5]/8);
          char ch[100];
          sprintf(ch,"%s %.2f",labelname[bboxs[i][0]-1], bboxs[i][5]*1.0/100);
          std::string temp(ch);
          cv::putText(image,temp,point1,CV_FONT_HERSHEY_COMPLEX,0.4,cv::Scalar(255,255,255));
      }
  }
  //输出结果，画框。。
  cv::imshow("yolo",image);

  std::string save_name(argv[3]);
  save_name = save_name.substr(0, save_name.find('.'));
  save_name = save_name + "_yolo_detect.jpg";
  std::cout << save_name << std::endl;
  cv::imwrite(save_name, image);

  cv::waitKey(0);
  //下面这三行注释的代码的作用，当你只有txt存储的weight的时候，需要将weight转化为caffemodel的时候就使用这三行代码。
  //caffe::NetParameter net_param;
 // net->ToProto(&net_param,false);
  //WriteProtoToBinaryFile(net_param, "/home/yang/yolo2caffe/yolo.caffemodel");
  return 1;
}
void loadweights(boost::shared_ptr<Net<float> >& net,char* argv){
  char txt_name[200];
  strcat(txt_name,argv);
  char path[200];
  const std::vector<boost::shared_ptr<Layer<float> > > layers = net->layers();
  int convolution_n = 0;
  int connect_n = 0;
  FILE* fp;
  char* name = (char*)malloc(sizeof(char)*100);
  boost::shared_ptr<Layer<float> > layer;
  std::vector<boost::shared_ptr<Blob<float> > > blobs;
  for(int i = 0; i < layers.size(); ++i){
    layer = layers[i];
    blobs = layer->blobs();
    if(layer->type() == std::string("Convolution")){
        ++convolution_n;
        std::cout << "convolution" << convolution_n <<std::endl;
        sprintf(path,"%s/convolution%d.txt",argv,convolution_n);
        //std::cout << path << std::endl;
        //sprintf(name,"/home/yang/yolo2caffe/yolo/yolo_convolution%d.txt",convolution_n);
        fp = fopen(path,"r");
        fread(blobs[1]->mutable_cpu_data(), sizeof(float), blobs[1]->count(), fp);
        fread(blobs[0]->mutable_cpu_data(), sizeof(float), blobs[0]->count(), fp);
    }
    else if(layer->type() == std::string("InnerProduct")){
        ++connect_n;
        std::cout << "Connect" << connect_n <<std::endl;
        sprintf(path,"%s/connect%d.txt",argv,connect_n);
        //std::cout << path << std::endl;
        fp = fopen(path,"r");
        fread(blobs[1]->mutable_cpu_data(), sizeof(float), blobs[1]->count(), fp);
        fread(blobs[0]->mutable_cpu_data(), sizeof(float), blobs[0]->count(), fp);
      }
  }
  if(fp != NULL)
    fclose(fp);
  delete []name;
}
void loaddata(boost::shared_ptr<Net<float> >& net, std::string image_path){
  Blob<float>* input_layer = net->input_blobs()[0];
  int width, height;
  width = input_layer->width();
  height = input_layer->height();
  int size = width*height;
  cv::Mat image = cv::imread(image_path,-1);
  cv::Mat image_resized;
  cv::resize(image, image_resized, cv::Size(height, width));
  float* input_data = input_layer->mutable_cpu_data();
  int temp,idx;
  for(int i = 0; i < height; ++i){
    uchar* pdata = image_resized.ptr<uchar>(i);
    for(int j = 0; j < width; ++j){
      temp = 3*j;
      idx = i*width+j;
      input_data[idx] = (pdata[temp+2]/127.5)-1;
      input_data[idx+size] = (pdata[temp+1]/127.5)-1;
      input_data[idx+2*size] = (pdata[temp+0]/127.5)-1;
    }
  }
  //cv::imshow("image",image_resized);
}
void getbox(std::vector<float> result,float* pro_obj,int* idx_class,std::vector<std::vector<int> >& bboxs,float thresh,cv::Mat image){
  float pro_class[121];
  int idx;
  float max_idx;
  float max;
  for(int i = 0; i < 11; ++i){
    for(int j = 0; j < 11;++j){
      max = 0;
      max_idx = 0;
      idx = 1*(i*11+j);
      for(int k = 0; k < 1; ++k){
        if (result[idx+k] > max){
          max = result[idx+k];
          max_idx = k+1;
        }
      }
      idx_class[i*11+j] = max_idx;
      pro_class[i*11+j] = max;
      pro_obj[(i*11+j)*2] = max*result[11*11*1+(i*11+j)*2];
      pro_obj[(i*11+j)*2+1] = max*result[11*11*1+(i*11+j)*2+1];
    }
  }
  std::vector<int> bbox;
  int x_min,x_max,y_min,y_max;
  float x,y,w,h;
  for(int i = 0; i < 11;++i){
    for(int j = 0; j < 11;++j){
      for(int k = 0; k < 2; ++k){
          if(pro_obj[(i*11+j)*2 + k] > thresh){
              //std::cout << "(" << i << "," << j << "," << k << ")" << " prob="<<pro_obj[(i*7+j)*2 + k] << " class="<<idx_class[i*7+j]<<std::endl;
              idx = 121*1 + 121*2 + ((i*11+j)*2+k)*4;
              x = image.cols*(result[idx++]+j)/11;
              y = image.rows*(result[idx++]+i)/11;
              w = image.cols*result[idx]*result[idx++];
              h = image.rows*result[idx]*result[idx];
              //std::cout << x <<" "<< y << " " << w <<" "<< h <<std::endl;
              x_min = x - w/2;
              y_min = y - h/2;
              x_max = x + w/2;
              y_max = y + h/2;
              bbox.clear();
              bbox.push_back(idx_class[i*7+j]);
              bbox.push_back(x_min);
              bbox.push_back(y_min);
              bbox.push_back(x_max);
              bbox.push_back(y_max);
              bbox.push_back(int(pro_obj[(i*11+j)*2 + k]*100));
              bboxs.push_back(bbox);
          }
      }
    }
  }
}
