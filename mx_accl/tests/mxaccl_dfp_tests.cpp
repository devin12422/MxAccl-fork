#include <gtest/gtest.h>
#include <fstream>
#include "memx/accl/utils/path.h"
#include "memx/accl/MxAccl.h"
#include "memx/accl/MxAcclMT.h"

namespace fs = std::filesystem;

fs::path mx_accl_path = MX::Utils::mx_get_accl_dir();
fs::path models_path = mx_accl_path/"tests"/"models";
fs::path dfp_path = models_path/"cascadePlus";
fs::path prepost_path = models_path/"plugin_models";


bool input_callback(std::vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){return false;}

bool output_callback(std::vector<const MX::Types::FeatureMap<float>*> src, int stream_id){return true;}

TEST(accl_dfp_tests, num_models_1){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_models(),1);
    accl.stop();
}

TEST(accl_dfp_tests, num_models_2){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_models(),1);
    accl.stop();
}

TEST(accl_dfp_tests, num_models_3){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_models(),2);
    accl.stop();
}

TEST(accl_manual_dfp_tests, num_models_1){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_models(),1);
}

TEST(accl_manual_dfp_tests, num_models_2){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_models(),2);
}

TEST(accl_dfp_tests, num_streams_1){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_streams(),0);
    accl.stop();
}

TEST(accl_dfp_tests, num_streams_2){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback,&output_callback,0);
    accl.start();
    accl.wait(); 
    GTEST_ASSERT_EQ(accl.get_num_streams(),1);   
    accl.stop(); 
}

TEST(accl_dfp_tests, num_streams_3){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback,&output_callback,0);
    accl.connect_stream(&input_callback,&output_callback,1);
    accl.connect_stream(&input_callback,&output_callback,2);
    accl.start();
    GTEST_ASSERT_EQ(accl.get_num_streams(),3);   
    accl.wait(); 
    accl.stop();
}

TEST(accl_dfp_tests, num_streams_4){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback,&output_callback,0);
    accl.connect_stream(&input_callback,&output_callback,1,1);
    accl.start();
    accl.wait(); 
    GTEST_ASSERT_EQ(accl.get_num_streams(),2);    
    accl.stop();
}

TEST(accl_dfp_tests,three_chip_dfp) {
    fs::path model_path = dfp_path/"mobilenet_3chip.dfp";
    MX::Runtime::MxAccl accl;
    try
    {
        accl.connect_dfp(model_path);
    }
    catch(std::runtime_error const & err)
    {
        EXPECT_EQ(err.what(),std::string("this dfp is made for 3 but only 4 are available on device 0"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected this dfp is made for 3 but only 4 are available on device 0";
    }
}

TEST(accl_dfp_tests, model_info_tests){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    MX::Types::MxModelInfo model_info = accl.get_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input_1");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "predictions");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 3);
    
    model_info = accl.get_model_info(1);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input_1");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "predictions");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 3);

    const char* expected_exception = "std::exception";
    try {
        model_info = accl.get_model_info(2);
    }
    catch(std::exception err) {
        std::cout<<std::string(err.what());
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
 
    accl.stop();

}

TEST(accl_dfp_tests, prepost_onnx_model_info_tests){
    fs::path model_path = dfp_path/"prepost_onnx.dfp";
    fs::path pre_model_path = prepost_path/"onnx"/"prepost_pre.onnx";
    fs::path post_model_path = prepost_path/"onnx"/"prepost_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_pre_model(pre_model_path);
    accl.connect_post_model(post_model_path);
    MX::Types::MxModelInfo model_info = accl.get_pre_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "input.1");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 3);
    
    model_info = accl.get_post_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "onnx::Mul_4");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "output");
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][1], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][2], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][3], 2);

    const char* expected_exception = "std::exception";
    try {
        model_info = accl.get_pre_model_info(2);
    }
    catch(std::exception err) {
        std::cout<<std::string(err.what());
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
    accl.stop();
}

TEST(accl_dfp_tests, prepost_tflite_model_info_tests){
    fs::path model_path = dfp_path/"prepost_tflite.dfp";
    fs::path pre_model_path = prepost_path/"tflite"/"prepost_pre.tflite";
    fs::path post_model_path = prepost_path/"tflite"/"prepost_post.tflite";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_pre_model(pre_model_path);
    accl.connect_post_model(post_model_path);
    MX::Types::MxModelInfo model_info = accl.get_pre_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "serving_default_input_1:0");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "pre_post/add/add");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 3);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 2);
    
    model_info = accl.get_post_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "pre_post/conv2d/Conv2D2");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "StatefulPartitionedCall:0");
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][1], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][2], 2);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][3], 1);

    const char* expected_exception = "std::exception";
    try {
        model_info = accl.get_post_model_info(2);
    }
    catch(std::exception err) {
        std::cout<<std::string(err.what());
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
 
    accl.stop();
}

TEST(accl_dfp_tests, prepost_tf_model_info_tests){
    fs::path model_path = dfp_path/"prepost_tf.dfp";
    fs::path pre_model_path = prepost_path/"tensorflow"/"prepost_pre.pb";
    fs::path post_model_path = prepost_path/"tensorflow"/"prepost_post.pb";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_pre_model(pre_model_path);
    accl.connect_post_model(post_model_path);
    MX::Types::MxModelInfo model_info = accl.get_pre_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input_1");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "StatefulPartitionedCall/pre_post/add");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 3);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 2);
    
    model_info = accl.get_post_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "StatefulPartitionedCall/pre_post/conv2d/Conv2D");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "Identity");
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][1], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][2], 2);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][3], 1);

    accl.stop();
}

TEST(accl_manual_dfp_tests, model_info_tests){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    MX::Types::MxModelInfo model_info = accl.get_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input_1");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "predictions");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 3);
    
    model_info = accl.get_model_info(1);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input_1");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "predictions");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 224);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 3);

    const char* expected_exception = "std::exception";
    try {
        model_info = accl.get_model_info(2);
    }
    catch(std::exception err) {
        std::cout<<std::string(err.what());
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
}

TEST(accl_manual_dfp_tests, prepost_onnx_model_info_tests){
    fs::path model_path = dfp_path/"prepost_onnx.dfp";
    fs::path pre_model_path = prepost_path/"onnx"/"prepost_pre.onnx";
    fs::path post_model_path = prepost_path/"onnx"/"prepost_post.onnx";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    accl.connect_pre_model(pre_model_path);
    accl.connect_post_model(post_model_path);
    MX::Types::MxModelInfo model_info = accl.get_pre_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "input.1");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 3);
    
    model_info = accl.get_post_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "onnx::Mul_4");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "output");
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][1], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][2], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][3], 2);

    const char* expected_exception = "std::exception";
    try {
        model_info = accl.get_pre_model_info(2);
    }
    catch(std::exception err) {
        std::cout<<std::string(err.what());
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
}

TEST(accl_manual_dfp_tests, prepost_tflite_model_info_tests){
    fs::path model_path = dfp_path/"prepost_tflite.dfp";
    fs::path pre_model_path = prepost_path/"tflite"/"prepost_pre.tflite";
    fs::path post_model_path = prepost_path/"tflite"/"prepost_post.tflite";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    accl.connect_pre_model(pre_model_path);
    accl.connect_post_model(post_model_path);
    MX::Types::MxModelInfo model_info = accl.get_pre_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "serving_default_input_1:0");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "pre_post/add/add");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 3);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 2);
    
    model_info = accl.get_post_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "pre_post/conv2d/Conv2D2");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "StatefulPartitionedCall:0");
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][1], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][2], 2);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][3], 1);

    const char* expected_exception = "std::exception";
    try {
        model_info = accl.get_post_model_info(2);
    }
    catch(std::exception err) {
        std::cout<<std::string(err.what());
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
}

TEST(accl_manual_dfp_tests, prepost_tf_model_info_tests){
    fs::path model_path = dfp_path/"prepost_tf.dfp";
    fs::path pre_model_path = prepost_path/"tensorflow"/"prepost_pre.pb";
    fs::path post_model_path = prepost_path/"tensorflow"/"prepost_post.pb";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    accl.connect_pre_model(pre_model_path);
    accl.connect_post_model(post_model_path);
    MX::Types::MxModelInfo model_info = accl.get_pre_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input_1");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "StatefulPartitionedCall/pre_post/add");
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 2);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 3);
    GTEST_ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 2);
    
    model_info = accl.get_post_model_info(0);
    GTEST_ASSERT_EQ(model_info.num_in_featuremaps, 1);
    GTEST_ASSERT_EQ(model_info.num_out_featuremaps, 1);
    GTEST_ASSERT_EQ(std::string(model_info.input_layer_names[0]), "StatefulPartitionedCall/pre_post/conv2d/Conv2D");
    GTEST_ASSERT_EQ(std::string(model_info.output_layer_names[0]), "Identity");
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][0], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][1], 1);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][2], 2);
    GTEST_ASSERT_EQ(model_info.out_featuremap_shapes[0][3], 1);
}

TEST(accl_dfp_tests, dfpv5_file){
    fs::path model_path = dfp_path/"mobilenet_dfpv5.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_models(),1);
    accl.stop();
}

TEST(accl_dfp_tests, dfpv6_file){
    fs::path model_path = dfp_path/"mobilenet_dfpv6.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    GTEST_ASSERT_EQ(accl.get_num_models(),1);
    accl.stop();
}

TEST(accl_dfp_tests, dfpv5_bytes){
    fs::path model_path = dfp_path/"mobilenet_dfpv5.dfp";
    std::ifstream fd(model_path);

    fd.seekg(0, std::ios::end);
    size_t leng = fd.tellg();
    fd.seekg(0, std::ios::beg);
    char *buffer = new char[leng];
    memset(buffer, 0, leng);

    fd.read(buffer, leng);
    MX::Runtime::MxAccl accl;
    accl.connect_dfp((uint8_t*) buffer);
    GTEST_ASSERT_EQ(accl.get_num_models(),1);
    accl.stop();
    delete [] buffer;
}

TEST(accl_dfp_tests, dfpv6_bytes){
    fs::path model_path = dfp_path/"mobilenet_dfpv6.dfp";
    std::ifstream fd(model_path);

    fd.seekg(0, std::ios::end);
    size_t leng = fd.tellg();
    fd.seekg(0, std::ios::beg);
    char *buffer = new char[leng];
    memset(buffer, 0, leng);

    fd.read(buffer, leng);
    MX::Runtime::MxAccl accl;
    accl.connect_dfp((uint8_t*) buffer);
    GTEST_ASSERT_EQ(accl.get_num_models(),1);
    accl.stop();
    delete [] buffer;
}

TEST(accl_dfp_tests, name_mathing_post_model_info){
    fs::path model_path = dfp_path/"name_matching_onnx.dfp";
    fs::path post_path = prepost_path/"onnx"/"name_matching_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_post_model(post_path);
    MX::Types::MxModelInfo model_info = accl.get_post_model_info(0);
    ASSERT_EQ(model_info.num_out_featuremaps, 2);
    ASSERT_EQ(std::string(model_info.output_layer_names[0]), "output");
    ASSERT_EQ(std::string(model_info.output_layer_names[1]), "output1");
    ASSERT_EQ(model_info.out_featuremap_shapes[0][0], 1);
    ASSERT_EQ(model_info.out_featuremap_shapes[0][1], 1);
    ASSERT_EQ(model_info.out_featuremap_shapes[0][2], 1);
    ASSERT_EQ(model_info.out_featuremap_shapes[0][3], 2);
    ASSERT_EQ(model_info.out_featuremap_shapes[1][0], 1);
    ASSERT_EQ(model_info.out_featuremap_shapes[1][1], 1);
    ASSERT_EQ(model_info.out_featuremap_shapes[1][2], 1);
    ASSERT_EQ(model_info.out_featuremap_shapes[1][3], 2);
    ASSERT_EQ(model_info.out_featuremap_sizes[0],2);
    ASSERT_EQ(model_info.out_featuremap_sizes[1],2);
}

TEST(accl_dfp_tests, name_mathing_model_info){
    fs::path model_path = dfp_path/"name_matching_pre_onnx.dfp";
    fs::path pre_path = prepost_path/"onnx"/"name_matching_pre.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_pre_model(pre_path);

    MX::Types::MxModelInfo model_info = accl.get_pre_model_info(0);
    ASSERT_EQ(model_info.num_in_featuremaps, 2);
    ASSERT_EQ(std::string(model_info.input_layer_names[0]), "input");
    ASSERT_EQ(std::string(model_info.input_layer_names[1]), "input1");
    ASSERT_EQ(model_info.in_featuremap_shapes[0][0], 1);
    ASSERT_EQ(model_info.in_featuremap_shapes[0][1], 2);
    ASSERT_EQ(model_info.in_featuremap_shapes[0][2], 2);
    ASSERT_EQ(model_info.in_featuremap_shapes[0][3], 3);
    ASSERT_EQ(model_info.in_featuremap_shapes[1][0], 1);
    ASSERT_EQ(model_info.in_featuremap_shapes[1][1], 2);
    ASSERT_EQ(model_info.in_featuremap_shapes[1][2], 2);
    ASSERT_EQ(model_info.in_featuremap_shapes[1][3], 3);
    ASSERT_EQ(model_info.in_featuremap_sizes[0],12);
    ASSERT_EQ(model_info.in_featuremap_sizes[1],12);
}