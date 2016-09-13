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

void loadweights(boost::shared_ptr<Net<float> >& net,char* argv);

int main(int argc, char** argv){
  //boost::progress_timer t;
Caffe::set_mode(Caffe::GPU);
  boost::shared_ptr<Net<float> > net(new Net<float>(argv[1], caffe::TEST));
  loadweights(net,argv[2]);//这行代码是还没caffemodel的时候使用的。
  //net->CopyTrainedLayersFromBinaryProto(argv[2]);//有caffemodel的时候就可以使用这行代码，跟上面一行互补使用。

  //下面这三行注释的代码的作用，当你只有txt存储的weight的时候，需要将weight转化为caffemodel的时候就使用这三行代码。
  caffe::NetParameter net_param;
  net->ToProto(&net_param,false);
  WriteProtoToBinaryFile(net_param,  argv[3]);
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
        std::cout << path << std::endl;
        fp = fopen(path,"r");
        fread(blobs[1]->mutable_cpu_data(), sizeof(float), blobs[1]->count(), fp);
        fread(blobs[0]->mutable_cpu_data(), sizeof(float), blobs[0]->count(), fp);
    }
    else if(layer->type() == std::string("InnerProduct")){
        ++connect_n;
        std::cout << "Connect" << connect_n <<std::endl;
        sprintf(path,"%s/connect%d.txt",argv,connect_n);
        std::cout << path << std::endl;
        fp = fopen(path,"r");
        fread(blobs[1]->mutable_cpu_data(), sizeof(float), blobs[1]->count(), fp);
        fread(blobs[0]->mutable_cpu_data(), sizeof(float), blobs[0]->count(), fp);
      }
  }
  if(fp != NULL)
    fclose(fp);
  delete []name;
}
