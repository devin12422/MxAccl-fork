#include <gtest/gtest.h>
#include <memx/accl/utils/path.h>
#include <memx/accl/MxAccl.h>
#include <dlfcn.h>

#define NUM_TEST_FRAMES 20
using namespace std;
namespace fs = std::filesystem;

fs::path mx_accl_path = MX::Utils::mx_get_accl_dir();
fs::path dfp_path = mx_accl_path/"tests"/"models"/"cascadePlus";

fs::path onnx_plugin_path = mx_accl_path/"tests"/"models/plugin_models/onnx";
fs::path tf_plugin_path = mx_accl_path/"tests"/"models/plugin_models/tensorflow";
fs::path tflite_plugin_path = mx_accl_path/"tests"/"models/plugin_models/tflite";

atomic_int  sent_num_frames_1 = 0;
atomic_int  recv_num_frames_1 = 0;
atomic_int  sent_num_frames_2 = 0;
atomic_int  recv_num_frames_2 = 0;

void init_num_frames(){
    sent_num_frames_1 = 0;
    recv_num_frames_1 = 0;
    sent_num_frames_2 = 0;
    recv_num_frames_2 = 0;
}


void test_num_frames(){
    EXPECT_EQ(sent_num_frames_1.load(),recv_num_frames_1.load());
    EXPECT_EQ(sent_num_frames_2.load(),recv_num_frames_2.load());
    EXPECT_EQ(recv_num_frames_1.load(),NUM_TEST_FRAMES);
}

bool input_callback_1(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input = {1,2,4,5,2,3,1,3,4,2,1,6};
    dst[0]->set_data(input.data());
    sent_num_frames_1++;
    return true;
}


bool output_callback_1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(170.0,output[0]);
    EXPECT_EQ(212.0,output[1]);
    recv_num_frames_1++;
    return true;
}

bool input_callback_2(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_2.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input = {5,1,7,2,4,3,5,1,2,6,0,2};
    dst[0]->set_data(input.data());
    sent_num_frames_2++;
    return true;
}

bool output_callback_2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(188.0,output[0]);
    EXPECT_EQ(184.0,output[1]);
    recv_num_frames_2++;
    return true;
}

bool input_callback_nopre_1(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input = {1,1,2,3,4,4,5,2,2,1,3,6};
    dst[0]->set_data(input.data());
    sent_num_frames_1++;
    return true;
}

bool input_callback_nopre_2(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_2.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input = {5,5,1,1,7,2,2,6,4,0,3,2};
    dst[0]->set_data(input.data());
    sent_num_frames_2++;
    return true;
}

bool output_callback_no_post_1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(85.0,output[0]);
    EXPECT_EQ(106.0,output[1]);
    recv_num_frames_1++;
    return true;
}

bool output_callback_no_post_2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(94.0,output[0]);
    EXPECT_EQ(92.0,output[1]);
    recv_num_frames_2++;
    return true;
}

bool output_callback_no_pre_1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(90.0,output[0]);
    EXPECT_EQ(132.0,output[1]);
    recv_num_frames_1++;
    return true;
}

bool output_callback_no_pre_2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(108.0,output[0]);
    EXPECT_EQ(104.0,output[1]);
    recv_num_frames_2++;
    return true;
}

TEST(accl_prepost_tests, onnx_1){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_onnx.dfp";
    fs::path pre_path = onnx_plugin_path/"prepost_pre.onnx";
    fs::path post_path = onnx_plugin_path/"prepost_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, onnx_2){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_onnx.dfp";
    fs::path pre_path = onnx_plugin_path/"prepost_pre.onnx";
    fs::path post_path = onnx_plugin_path/"prepost_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,1);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, onnx_3){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_onnx_multimodel.dfp";
    fs::path pre_path = onnx_plugin_path/"prepost_pre.onnx";
    fs::path post_path = onnx_plugin_path/"prepost_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,1,1);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.connect_post_model(post_path,1);
    accl.connect_pre_model(pre_path,1);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, onnx_4){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_onnx_multimodel.dfp";
    fs::path pre_path = onnx_plugin_path/"prepost_pre.onnx";
    fs::path post_path = onnx_plugin_path/"prepost_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_1,&output_callback_1,1,1);
    accl.connect_stream(&input_callback_2,&output_callback_2,2,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,3,1);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.connect_post_model(post_path,1);
    accl.connect_pre_model(pre_path,1);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, onnx_no_post){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_onnx.dfp";
    fs::path pre_path = onnx_plugin_path/"prepost_pre.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_2,&output_callback_no_post_2,1);
    accl.connect_stream(&input_callback_1,&output_callback_no_post_1,0);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}


TEST(accl_prepost_tests, onnx_no_pre){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_onnx.dfp";
    fs::path post_path = onnx_plugin_path/"prepost_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_nopre_2,&output_callback_no_pre_2,1);
    accl.connect_stream(&input_callback_nopre_1,&output_callback_no_pre_1,0);
    accl.connect_post_model(post_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

bool input_callback_tf1(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input = {1,1,2,3,4,4,5,2,2,1,3,6};
    dst[0]->set_data(input.data());
    sent_num_frames_1++;
    return true;
}

bool input_callback_tf2(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_2.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input = {5,5,1,1,7,2,2,6,4,0,3,2};
    dst[0]->set_data(input.data());
    sent_num_frames_2++;
    return true;
}

TEST(accl_prepost_tests, tensorflow_1){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tf.dfp";
    fs::path pre_path = tf_plugin_path/"prepost_pre.pb";
    fs::path post_path = tf_plugin_path/"prepost_post.pb";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_1,0);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tensorflow_2){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tf.dfp";
    fs::path pre_path = tf_plugin_path/"prepost_pre.pb";
    fs::path post_path = tf_plugin_path/"prepost_post.pb";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_2,1);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tensorflow_3){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tf_multimodel.dfp";
    fs::path pre_path = tf_plugin_path/"prepost_pre.pb";
    fs::path post_path = tf_plugin_path/"prepost_post.pb";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_2,1,1);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.connect_post_model(post_path,1);
    accl.connect_pre_model(pre_path,1);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tensorflow_4){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tf_multimodel.dfp";
    fs::path pre_path = tf_plugin_path/"prepost_pre.pb";
    fs::path post_path = tf_plugin_path/"prepost_post.pb";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_2,1,1);
    accl.connect_stream(&input_callback_tf1,&output_callback_1,2,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_2,3,1);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.connect_post_model(post_path,1);
    accl.connect_pre_model(pre_path,1);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tensorflow_no_pre){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tf.dfp";
    fs::path post_path = tf_plugin_path/"prepost_post.pb";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_no_pre_1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_no_pre_2,1);
    accl.connect_post_model(post_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tensorflow_no_post){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tf.dfp";
    fs::path pre_path = tf_plugin_path/"prepost_pre.pb";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_no_post_1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_no_post_2,1);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

bool output_callback_tflite1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(95.0,output[0]);
    EXPECT_EQ(116.0,output[1]);
    recv_num_frames_1++;
    return true;
}

bool output_callback_tflite2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(104.0,output[0]);
    EXPECT_EQ(102.0,output[1]);
    recv_num_frames_2++;
    return true;
}

bool output_callback_tflitenopre1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(55.0,output[0]);
    EXPECT_EQ(76.0,output[1]);
    recv_num_frames_1++;
    return true;
}

bool output_callback_tflitenopre2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(64.0,output[0]);
    EXPECT_EQ(62.0,output[1]);
    recv_num_frames_2++;
    return true;
}

bool output_callback_tflitenopost1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(85.0,output[0]);
    EXPECT_EQ(106.0,output[1]);
    recv_num_frames_1++;
    return true;
}

bool output_callback_tflitenopost2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output = {0,0};
    src[0]->get_data(output.data());
    EXPECT_EQ(94.0,output[0]);
    EXPECT_EQ(92.0,output[1]);
    recv_num_frames_2++;
    return true;
}

TEST(accl_prepost_tests, tflite_1){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tflite.dfp";
    fs::path pre_path = tflite_plugin_path/"prepost_pre.tflite";
    fs::path post_path = tflite_plugin_path/"prepost_post.tflite";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_tflite1,0);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tflite_3){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tflite_multimodel.dfp";
    fs::path pre_path = tflite_plugin_path/"prepost_pre.tflite";
    fs::path post_path = tflite_plugin_path/"prepost_post.tflite";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_tflite1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_tflite2,1,1);
    accl.connect_stream(&input_callback_tf1,&output_callback_tflite1,2,1);
    accl.connect_stream(&input_callback_tf2,&output_callback_tflite2,3);
    accl.connect_post_model(post_path);
    accl.connect_pre_model(pre_path);
    accl.connect_post_model(post_path,1);
    accl.connect_pre_model(pre_path,1);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tflite_nopre){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tflite.dfp";
    fs::path pre_path = tflite_plugin_path/"prepost_pre.tflite";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_tflitenopost1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_tflitenopost2,1);
    accl.connect_pre_model(pre_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_prepost_tests, tflite_nopost){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tflite.dfp";
    fs::path post_path = tflite_plugin_path/"prepost_post.tflite";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_tf1,&output_callback_tflitenopre1,0);
    accl.connect_stream(&input_callback_tf2,&output_callback_tflitenopre2,1);
    accl.connect_post_model(post_path);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

bool name_matching_input_callback(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input1 = {5,1,7,2,4,3,5,1,2,6,0,2};
    dst[0]->set_data(input1.data(),true);
    sent_num_frames_1++;
    return true;
}

bool name_matching_output_callback(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output1 = {0,0};
    std::vector<float> output2 = {1,1};
    src[0]->get_data(output1.data());
    src[1]->get_data(output2.data());
    EXPECT_EQ(54.0,output2[0]);
    EXPECT_EQ(52.0,output2[1]);
    EXPECT_EQ(108.0,output1[0]);
    EXPECT_EQ(104.0,output1[1]);
    recv_num_frames_1++;
    return true;
}

TEST(accl_prepost_tests, name_matching_onnx){
    init_num_frames();
    fs::path model_path = dfp_path/"name_matching_onnx.dfp";
    fs::path post_path = onnx_plugin_path/"name_matching_post.onnx";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_post_model(post_path);
    accl.connect_stream(&name_matching_input_callback,&name_matching_output_callback,0);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();    
}

bool name_matching_tflite_input_callback(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=NUM_TEST_FRAMES)
    return false;
    std::vector<float> input1 = {5,5,1,1,7,2,2,6,4,0,3,2};
    dst[0]->set_data(input1.data());
    sent_num_frames_1++;
    return true;
}

bool name_matching_tflite_output_callback(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    std::vector<float> output1 = {0,0};
    std::vector<float> output2 = {1,1};
    src[0]->get_data(output1.data());
    src[1]->get_data(output2.data());
    EXPECT_EQ(54.0,output2[0]);
    EXPECT_EQ(52.0,output2[1]);
    EXPECT_EQ(85.0,output1[0]);
    EXPECT_EQ(70.0,output1[1]);
    recv_num_frames_1++;
    return true;
}

TEST(accl_prepost_tests, name_matching_tflite){
    init_num_frames();
    fs::path model_path = dfp_path/"name_matching_tflite.dfp";
    fs::path post_path = tflite_plugin_path/"name_matching_post.tflite";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_post_model(post_path);
    accl.connect_stream(&name_matching_tflite_input_callback,&name_matching_tflite_output_callback,0);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();    
}
