#include <gtest/gtest.h>
#include "memx/accl/prepost.h"
#include "memx/accl/utils/featureMap.h"
namespace fs = std::filesystem;

TEST(accl_utility_tests, split_func){
    std::vector<std::string> split_vec = prepost_split("first:second:third");
    ASSERT_EQ("first/",split_vec[0]);
    ASSERT_EQ("second/",split_vec[1]);
    ASSERT_EQ("third/",split_vec[2]);
}

TEST(accl_utility_tests, createObject_func_fail_plugin){
    std::vector<size_t> sizes{0};
    const char* expected_exception = "Failed to load shared object: libunknown.so. Try to reinstall memx-accl";
    try {
        PrePost* obj = createObject("libunknown","createOnnx","", sizes,Plugin_Onnx);
    }
    catch(std::runtime_error err) {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
}

TEST(accl_utility_tests, createObject_func_fail_createFunc){
    std::vector<size_t> sizes{0};
    const char* expected_exception = "Couldn't load the function: createUnknown";
    try {
        PrePost* obj = createObject("libonnxinfer","createUnknown","", sizes,Plugin_Onnx);
    }
    catch(std::runtime_error err) {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }
}

TEST(accl_utility_tests, featuremap_size_constructor){
    MX::Types::FeatureMap<float> fmp(1000);
    ASSERT_EQ(4000,fmp.get_formatted_size());

    MX::Types::FeatureMap<float> fmp1(1000,MX::Types::MX_FMT_BF16);
    ASSERT_EQ(2000,fmp1.get_formatted_size());

    MX::Types::FeatureMap<float> fmp5(1000,MX::Types::MX_FMT_GBF80,1,1,1,1);
    ASSERT_EQ(10000,fmp5.get_formatted_size());

    MX::Types::FeatureMap<float> fmp6(1000,MX::Types::MX_FMT_GBF80_ROW,1,1,1,1);
    ASSERT_EQ(12,fmp6.get_formatted_size());

    // MX::Types::FeatureMap<uint8_t> fmp2(1000,MX::Types::MX_FMT_RGB888);
    // ASSERT_EQ(1000,fmp2.get_formatted_size());

    // const char* expected_exception = "featureMap given a removed format rgb565/yuv422/yuy2";
    // try {
    //     MX::Types::FeatureMap<uint8_t> fmp3(1000,MX::Types::MX_FMT_RGB565);
    // }
    // catch(std::runtime_error err) {
    //     EXPECT_EQ(std::string(err.what()),expected_exception);
    // }
    // catch(...) {
    //     FAIL() << "Expected :"<<expected_exception;
    // }  

    // expected_exception = "featureMap given a removed format rgb565/yuv422/yuy2";
    // try {
    //     MX::Types::FeatureMap<uint8_t> fmp4(1000,MX::Types::MX_FMT_YUV422);
    // }
    // catch(std::runtime_error err) {
    //     EXPECT_EQ(std::string(err.what()),expected_exception);
    // }
    // catch(...) {
    //     FAIL() << "Expected :"<<expected_exception;
    // }  

    // expected_exception = "featureMap<uint8_t> was given a float-type format";
    // try {
    //     MX::Types::FeatureMap<uint8_t> fmp7(1000,MX::Types::MX_FMT_BF16);
    // }
    // catch(std::runtime_error err) {
    //     EXPECT_EQ(std::string(err.what()),expected_exception);
    // }
    // catch(...) {
    //     FAIL() << "Expected :"<<expected_exception;
    // }         

    const char* expected_exception = "featureMap<float> was given RGB888 format";
    try {
        MX::Types::FeatureMap<float> fmp8(1000,MX::Types::MX_FMT_RGB888);
    }
    catch(std::runtime_error err) {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }   
}

TEST(accl_utility_tests, featuremap_data_constructor){
    float* data = new float[1000];
    MX::Types::FeatureMap<float> fmp(data,1000);
    ASSERT_EQ(4000,fmp.get_formatted_size());

    MX::Types::FeatureMap<float> fmp1(data,1000,MX::Types::MX_FMT_BF16);
    ASSERT_EQ(2000,fmp1.get_formatted_size());

    float* gbf_data = new float[1];
    MX::Types::FeatureMap<float> fmp5(gbf_data,1,MX::Types::MX_FMT_GBF80,1,1,1,1);
    ASSERT_EQ(10,fmp5.get_formatted_size());

    MX::Types::FeatureMap<float> fmp6(gbf_data,1,MX::Types::MX_FMT_GBF80_ROW,1,1,1,1);
    ASSERT_EQ(12,fmp6.get_formatted_size());

    // uint8_t* int_data = new uint8_t[1000];
    // MX::Types::FeatureMap<uint8_t> fmp2(int_data,1000,MX::Types::MX_FMT_RGB888);
    // ASSERT_EQ(1000,fmp2.get_formatted_size());

    // const char* expected_exception = "featureMap given a removed format rgb565/yuv422/yuy2";
    // try {
    //     MX::Types::FeatureMap<uint8_t> fmp3(int_data,1000,MX::Types::MX_FMT_RGB565);
    // }
    // catch(std::runtime_error err) {
    //     EXPECT_EQ(std::string(err.what()),expected_exception);
    // }
    // catch(...) {
    //     FAIL() << "Expected :"<<expected_exception;
    // }  

    // expected_exception = "featureMap given a removed format rgb565/yuv422/yuy2";
    // try {
    //     MX::Types::FeatureMap<uint8_t> fmp4(int_data,1000,MX::Types::MX_FMT_YUV422);
    // }
    // catch(std::runtime_error err) {
    //     EXPECT_EQ(std::string(err.what()),expected_exception);
    // }
    // catch(...) {
    //     FAIL() << "Expected :"<<expected_exception;
    // }  

    // expected_exception = "featureMap<uint8_t> was given a float-type format";
    // try {
    //     MX::Types::FeatureMap<uint8_t> fmp7(int_data,1000,MX::Types::MX_FMT_BF16);
    // }
    // catch(std::runtime_error err) {
    //     EXPECT_EQ(std::string(err.what()),expected_exception);
    // }
    // catch(...) {
    //     FAIL() << "Expected :"<<expected_exception;
    // }         

    const char* expected_exception = "featureMap<float> was given RGB888 format";
    try {
        MX::Types::FeatureMap<float> fmp8(data,1000,MX::Types::MX_FMT_RGB888);
    }
    catch(std::runtime_error err) {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }   

    delete[] data;
    // delete[] int_data;
    delete[] gbf_data;
}

TEST(accl_utility_tests, featuremap_copy_constructor){
    MX::Types::FeatureMap<float> fmp(1000,MX::Types::MX_FMT_BF16);
    MX::Types::FeatureMap<float> fmp_copy(2000);
    fmp_copy = fmp;
    ASSERT_EQ(2000,fmp.get_formatted_size());
    ASSERT_EQ(2000,fmp_copy.get_formatted_size());

    // MX::Types::FeatureMap<uint8_t> fmp_int(1000,MX::Types::MX_FMT_RGB888);
    // MX::Types::FeatureMap<uint8_t> fmp_int_copy(1000,MX::Types::MX_FMT_RGB888);
    // fmp_int = fmp_int;
    // ASSERT_EQ(1000,fmp_int.get_formatted_size());
    // ASSERT_EQ(1000,fmp_int_copy.get_formatted_size());
}

TEST(accl_utility_tests, featuremap_shape){
    MX::Types::FeatureMap<float> fmp(3*224*224,MX::Types::MX_FMT_BF16,224,224,1,3);
    ASSERT_EQ(301056,fmp.get_formatted_size());
    ASSERT_EQ(224,fmp.shape()[0]);
    ASSERT_EQ(224,fmp.shape()[1]);
    ASSERT_EQ(1,fmp.shape()[2]);
    ASSERT_EQ(3,fmp.shape()[3]);

    ASSERT_EQ(1,fmp.shape(true)[0]);
    ASSERT_EQ(3,fmp.shape(true)[1]);
    ASSERT_EQ(224,fmp.shape(true)[2]);
    ASSERT_EQ(224,fmp.shape(true)[3]);
}

TEST(accl_utility_tests, featuremap_data_len){
    MX::Types::FeatureMap<float> fmp(2);
    float send_data[2] = {1,3};
    float* return_data = new float[2];
    return_data[0] =0;
    return_data[1] =0;
    fmp.set_data_len(send_data,2);
    fmp.get_data_len(return_data,1);
    ASSERT_FLOAT_EQ(1,return_data[0]);
    ASSERT_FLOAT_EQ(0,return_data[1]);
}

TEST(accl_utility_tests, featuremap_data_len_zero){
    MX::Types::FeatureMap<float> fmp(2);
    float send_data[2] = {1,3};
    float* return_data = new float[2];
    fmp.set_data_len(send_data);
    fmp.get_data_len(return_data);
    ASSERT_FLOAT_EQ(1,return_data[0]);
    ASSERT_FLOAT_EQ(3,return_data[1]);
}

TEST(accl_utility_tests, mxtypes_data){
    MX::Types::ShapeVector shape(1,2,3,2);
    int64_t* data = shape.data();
    ASSERT_EQ(1,data[0]);
    ASSERT_EQ(2,data[1]);
    ASSERT_EQ(3,data[2]);
    ASSERT_EQ(2,data[3]);
}

TEST(accl_utility_tests, mxtypes_data_size){
    MX::Types::ShapeVector shape(1,2,3,2);
    ASSERT_EQ(4,shape.size());
}

TEST(accl_utility_tests, mxtypes_operator){
    MX::Types::ShapeVector shape(1,2,3,2);
    ASSERT_EQ(1,shape[0]);
    ASSERT_EQ(2,shape[1]);
    ASSERT_EQ(3,shape[2]);
    ASSERT_EQ(2,shape[3]);   

    const char* expected_exception = "Error: Index out of range.";
    try {
        int64_t shp = shape[4];
    }
    catch(std::runtime_error err) {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...) {
        FAIL() << "Expected :"<<expected_exception;
    }  
}

TEST(accl_utility_tests, post_pattern_matching){
    std::string plugin_path = "libonnxinfer";
    fs::path mx_home_path = MX::Utils::mx_get_home_dir();
    fs::path model_path = mx_home_path/"samples"/"models/yolov7Tiny416/model_0_yolov7-tiny_416_post.onnx";
    PrePost* obj = createObject(plugin_path,"createOnnx",model_path, {},Plugin_Onnx);
    obj->match_names({"/model/model.77/m.0/Conv_output_0","/model/model.77/m.2/Conv_output_0","/model/model.77/m.1/Conv_output_0","/model/model.77/m.3/Conv_output_0"},Process_Post);
    ASSERT_EQ(obj->dfp_pattern[0],0);
    ASSERT_EQ(obj->dfp_pattern[1],2);
    ASSERT_EQ(obj->dfp_pattern[2],1);
    ASSERT_EQ(obj->real_featuremaps.size(),1);
    ASSERT_EQ(obj->real_featuremaps[0],3);
    delete obj;
}

TEST(accl_utility_tests, post_num_mismatch){
    std::string plugin_path = "libonnxinfer";
    fs::path mx_home_path = MX::Utils::mx_get_home_dir();
    fs::path model_path = mx_home_path/"samples"/"models/yolov7Tiny416/model_0_yolov7-tiny_416_post.onnx";
    PrePost* obj = createObject(plugin_path,"createOnnx",model_path, {},Plugin_Onnx);
    try {
        obj->match_names({"/model/model.77/m.0/Conv_output_0","/model/model.77/m.2/Conv_output_0"},Process_Post);
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
        EXPECT_EQ(std::string(e.what()), "input size of post-processing model is greater than output size of dfp");
    } catch (...) {
        FAIL() << "Expected std::logic_error::input size of post-processing model is greater than output size of dfp";
    }
    delete obj;
}

TEST(accl_utility_tests, post_name_mismatch){
    std::string plugin_path = "libonnxinfer";
    fs::path mx_home_path = MX::Utils::mx_get_home_dir();
    fs::path model_path = mx_home_path/"samples"/"models/yolov7Tiny416/model_0_yolov7-tiny_416_post.onnx";
    PrePost* obj = createObject(plugin_path,"createOnnx",model_path, {},Plugin_Onnx);
    try {
        obj->match_names({"/model/model.77/m.0/Conv_output_0","/model/model.77/m.2/Conv_output_0","/model/model.77/m.3/Conv_output_0"},Process_Post);
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
        EXPECT_EQ(std::string(e.what()), "post-processing model input names don't match dfp output names");
    } catch (...) {
        FAIL() << "Expected std::logic_error::post-processing model input names don't match dfp output names";
    }
    delete obj;
}

TEST(accl_utility_tests, pre_pattern_matching){
    std::string plugin_path = "libtfliteinfer";
    fs::path mx_home_path = MX::Utils::mx_get_home_dir();
    fs::path model_path = mx_home_path/"mx_accl"/"tests/models/plugin_models/tflite/prepost_pre.tflite";
    PrePost* obj = createObject(plugin_path,"createTflite",model_path, {},Plugin_Tflite);
    obj->match_names({"pre_post/add/add1","pre_post/add/add"},Process_Pre);
    ASSERT_EQ(obj->dfp_pattern[0],1);
    ASSERT_EQ(obj->real_featuremaps[0],0);
    delete obj;
}

TEST(accl_utility_tests, pre_num_mismatch){
    std::string plugin_path = "libtfliteinfer";
    fs::path mx_home_path = MX::Utils::mx_get_home_dir();
    fs::path model_path = mx_home_path/"mx_accl"/"tests/models/plugin_models/tflite/prepost_pre.tflite";
    PrePost* obj = createObject(plugin_path,"createTflite",model_path, {},Plugin_Tflite);
    try {
        obj->match_names({},Process_Pre);
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
        EXPECT_EQ(std::string(e.what()), "output size of pre-processing model is greater than input size of dfp");
    } catch (...) {
        FAIL() << "Expected std::logic_error::output size of pre-processing model is greater than input size of dfp";
    }
    delete obj;
}

TEST(accl_utility_tests, pre_name_mismatch){
    std::string plugin_path = "libtfliteinfer";
    fs::path mx_home_path = MX::Utils::mx_get_home_dir();
    fs::path model_path = mx_home_path/"mx_accl"/"tests/models/plugin_models/tflite/prepost_pre.tflite";
    PrePost* obj = createObject(plugin_path,"createTflite",model_path, {},Plugin_Tflite);
    try {
        obj->match_names({"pre_post/add/add1"},Process_Pre);
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
        EXPECT_EQ(std::string(e.what()), "pre-processing model output names don't match dfp input names");
    } catch (...) {
        FAIL() << "Expected std::logic_error::pre-processing model output names don't match dfp input names";
    }
    delete obj;
}
