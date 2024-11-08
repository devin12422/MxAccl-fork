#include <gtest/gtest.h>
#include <numeric>
#include <thread>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include "string.h"
#include "memx/accl/utils/path.h"
#include "memx/accl/MxAccl.h"

using namespace std;
namespace fs = std::filesystem;

fs::path mx_accl_path = MX::Utils::mx_get_accl_dir();
fs::path dfp_path = mx_accl_path/"tests"/"models"/"cascadePlus";


vector<size_t> getTopNMaxIndices(const vector<float>& values, size_t n) {
    vector<size_t> indices(values.size());
    iota(indices.begin(), indices.end(), 0);

    // Use partial_sort to get the top N maximum indices
    partial_sort(indices.begin(), indices.begin() + n, indices.end(),
                      [&values](size_t i, size_t j) {
                          return values[i] > values[j];
                      });

    // Resize the vector to keep only the top N indices
    indices.resize(n);

    return indices;
}

cv::Mat mobilenet_load_image(const char* img_path){
    cv::Mat img = cv::imread(img_path);
    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(224, 224), cv::INTER_LINEAR);
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC3, 1.0/255.0, -1.0);
    return float_image;
}

vector<int> num_frames{10,30,50,20,10,30,50,20};
vector<int> recv_num_frames{0,0,0,0,0,0,0,0};
vector<string> images{"dog.png","strawberry.jpg"};

atomic_int  sent_num_frames_1 = 0;
atomic_int  recv_num_frames_1 = 0;
atomic_int  sent_num_frames_2 = 0;
atomic_int  recv_num_frames_2 = 0;
atomic_int  sent_num_frames_3 = 0;

void init_num_frames(){
    sent_num_frames_1 = 0;
    recv_num_frames_1 = 0;
    sent_num_frames_2 = 0;
    recv_num_frames_2 = 0;
    sent_num_frames_3 = 0;
}


void test_num_frames(){
    GTEST_ASSERT_EQ(sent_num_frames_1.load(),recv_num_frames_1.load());
    GTEST_ASSERT_EQ(sent_num_frames_2.load(),recv_num_frames_2.load());
}

bool input_callback_ff(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(num_frames[stream_id]<=0){
        return false;
    }
    fs::path img_path = mx_accl_path/"tests"/images[num_frames[stream_id]%2];
    cv::Mat input_image = mobilenet_load_image(img_path.c_str());
    dst[0]->set_data((float*)input_image.data);
    num_frames[stream_id]--;
    return true;
}

bool output_callback_ff(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float[1000];
    src[0]->get_data(output_1);
    vector<float> floatVector(output_1, output_1 + 1000);
    vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);
    if(recv_num_frames[stream_id]%2==0)
    EXPECT_EQ(235,top5[0]);
    else
    EXPECT_EQ(949,top5[0]);
    delete[] output_1;
    recv_num_frames[stream_id]++;
    return true;
}

bool input_callback_1(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=20)
    return false;
    fs::path img_path = mx_accl_path/"tests"/"dog.png";
    cv::Mat input_image = mobilenet_load_image(img_path.c_str());
    dst[0]->set_data((float*)input_image.data);
    sent_num_frames_1++;
    return true;
}

bool output_callback_1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float[1000];
    src[0]->get_data(output_1);
    vector<float> floatVector(output_1, output_1 + 1000);
    vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);
    
    // there might be some cascade vs. cascade+ difference here,
    // but it should be ~85% confident that this is a dog,
    // so the bottom 4 of top5 are insignficiant noise
    EXPECT_EQ(235,top5[0]);
    delete[] output_1;
    recv_num_frames_1++;
    // cout<<recv_num_frames[stream_id]<<endl;
    return true;
}

bool input_callback_2(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_2.load()>=20)
    return false;
    fs::path img_path = mx_accl_path/"tests"/"strawberry.jpg";
    cv::Mat input_image = mobilenet_load_image(img_path.c_str());
    dst[0]->set_data((float*)input_image.data);
    sent_num_frames_2++;
    return true;
}

bool output_callback_2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_2 = new float[1000];
    src[0]->get_data(output_2);
    vector<float> floatVector(output_2, output_2 + 1000);
    vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);

    // there might be some cascade vs. cascade+ difference here,
    // but it should be ~85% confident that this is a strawberry,
    // so the bottom 4 of top5 are insignficiant noise
    EXPECT_EQ(949,top5[0]);
    delete[] output_2;
    recv_num_frames_2++;
    return true;
}

TEST(accl_accuracy_test,mobilenet_ff){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_ff,&output_callback_ff,0);
    accl.connect_stream(&input_callback_ff,&output_callback_ff,1);
    accl.connect_stream(&input_callback_ff,&output_callback_ff,2,1);
    accl.connect_stream(&input_callback_ff,&output_callback_ff,3,1);
    accl.connect_stream(&input_callback_ff,&output_callback_ff,4);
    accl.connect_stream(&input_callback_ff,&output_callback_ff,5);
    accl.connect_stream(&input_callback_ff,&output_callback_ff,6,1);
    accl.connect_stream(&input_callback_ff,&output_callback_ff,7,1);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    ASSERT_EQ(recv_num_frames[0],10);
    ASSERT_EQ(recv_num_frames[1],30);
    ASSERT_EQ(recv_num_frames[2],50);
    ASSERT_EQ(recv_num_frames[3],20);
    ASSERT_EQ(recv_num_frames[4],10);
    ASSERT_EQ(recv_num_frames[5],30);
    ASSERT_EQ(recv_num_frames[6],50);
    ASSERT_EQ(recv_num_frames[7],20);
}

TEST(accl_accuracy_test,mobilenet_1){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_2){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_3){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,0,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_4){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,1);
    accl.connect_stream(&input_callback_1,&output_callback_1,2,1);
    accl.connect_stream(&input_callback_2,&output_callback_2,3,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_dataflow_tests, timed_stop){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_1,&output_callback_1,1);
    accl.connect_stream(&input_callback_1,&output_callback_1,2);
    accl.connect_stream(&input_callback_1,&output_callback_1,3);
    accl.start();
    std::this_thread::sleep_for(20ms);
    accl.stop();
}

#ifndef CASCADE
TEST(accl_accuracy_test,mobilenet_iFP_oFP){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_iFP_oFP.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_iBF_oFP){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_iBF_oFP.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_iBF_oBF){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_iBF_oBF.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_iFP_oBF){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_iFP_oBF.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_iGBF_oFP){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_iGBF_oFP.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_iFP_oGBF){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_iFP_oGBF.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_iGBF_oGBF){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_iGBF_oGBF.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}
#endif




// Helper functions to test shape transforms inside API
void transpose_4d(float* input, float* output,  int dim_h, int dim_w, int dim_z, int num_ch) {
    for (int c = 0; c < num_ch; ++c) {
        for (int h = 0; h < dim_h; ++h) {
            for (int w = 0; w < dim_w; ++w) {
                for (int d = 0; d < dim_z; ++d) {
                    output[c * dim_h * dim_w * dim_z + h * dim_w * dim_z + w * dim_z + d] =
                        input[h * dim_w * dim_z * num_ch + w * dim_z * num_ch + d * num_ch + c];
                }
            }
        }
    }
}


cv::Mat convert_to_chfirst(const cv::Mat& input_image) {
    // Reshape the image to 1x(h*w)*3
    cv::Mat reshaped_image = input_image.reshape(1, 1);
    cv::Mat reshaped_image1 = reshaped_image.clone();
    // Reshape it again to 1xhxwx3
    transpose_4d((float *)reshaped_image.data, (float *)reshaped_image1.data, input_image.rows, input_image.cols, 1, 3);
    return reshaped_image1;
}



bool input_callback_chfirst(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_2.load()>=20)
    return false;
    fs::path img_path = mx_accl_path/"tests"/"strawberry.jpg";
    cv::Mat input_image = mobilenet_load_image(img_path.c_str());
    cv::Mat transposedFrame = convert_to_chfirst(input_image);
    dst[0]->set_data((float*)transposedFrame.data, true); // marking channel first image is sent
    sent_num_frames_2++;
    return true;
}

bool output_callback_chfirst(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_2 = new float[1000];
    src[0]->get_data(output_2);
    vector<float> floatVector(output_2, output_2 + 1000);
    vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);
    //printf("top5: %lu, %lu, %lu, %lu, %lu\n", top5[0], top5[1], top5[2], top5[3], top5[4]);
    //printf("top5: [%f, %f, %f, %f, %f]\n", floatVector[top5[0]], floatVector[top5[1]], floatVector[top5[2]], floatVector[top5[3]], floatVector[top5[4]]);

    // there might be some cascade vs. cascade+ difference here,
    // but it should be ~85% confident that this is a strawberry,
    // so the bottom 4 of top5 are insignficiant noise
    EXPECT_EQ(949,top5[0]);
    delete[] output_2;
    recv_num_frames_2++;
    return true;
}

bool input_callback_chfirst1(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=20)
    return false;
    fs::path img_path = mx_accl_path/"tests"/"dog.png";
    cv::Mat input_image = mobilenet_load_image(img_path.c_str());
    cv::Mat transposedFrame = convert_to_chfirst(input_image);
    dst[0]->set_data((float*)transposedFrame.data, true); // marking channel first image is sent
    sent_num_frames_1++;
    return true;
}

bool output_callback_chfirst1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float[1000];
    src[0]->get_data(output_1);
    vector<float> floatVector(output_1, output_1 + 1000);
    vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);
    EXPECT_EQ(235,top5[0]);
    delete[] output_1;
    recv_num_frames_1++;

    return true;
}


bool input_callback_chfirstfail(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=20)
    return false;
    fs::path img_path = mx_accl_path/"tests"/"dog.png";
    cv::Mat input_image = mobilenet_load_image(img_path.c_str());
    cv::Mat transposedFrame = convert_to_chfirst(input_image);
    dst[0]->set_data((float*)transposedFrame.data, false); // marking channel first image is sent but calling normal copy
    sent_num_frames_1++;
    return true;
}

bool output_callback_chfirstfail(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float[1000];
    src[0]->get_data(output_1);
    vector<float> floatVector(output_1, output_1 + 1000);
    vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);
    EXPECT_NE(235,top5[0]); // since the data passed is not transformed the output should equate to expected output
    delete[] output_1;
    recv_num_frames_1++;

    return true;
}




// Testing channel first to last conversions

TEST(accl_accuracy_test,mobilenet_chfirst1){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_chfirst,&output_callback_chfirst,1);
    accl.connect_stream(&input_callback_chfirst1,&output_callback_chfirst1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_chfirst2){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_chfirst,&output_callback_chfirst,1,1);
    accl.connect_stream(&input_callback_chfirst1,&output_callback_chfirst1,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_chfirst3){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_chfirst,&output_callback_chfirst,1);
    accl.connect_stream(&input_callback_chfirst1,&output_callback_chfirst1,0);
    accl.connect_stream(&input_callback_chfirst,&output_callback_chfirst,3,1);
    accl.connect_stream(&input_callback_chfirst1,&output_callback_chfirst1,2,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test,mobilenet_chfirstfail){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_chfirstfail,&output_callback_chfirstfail,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_accuracy_test, mobilenet_workers){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_1,&output_callback_1,1);
    accl.connect_stream(&input_callback_1,&output_callback_1,2);
    accl.set_num_workers(1,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();    
}
