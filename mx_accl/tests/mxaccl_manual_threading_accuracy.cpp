#include <gtest/gtest.h>
#include <numeric>
#include <thread>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include "string.h"
#include "memx/accl/utils/path.h"
#include "memx/accl/MxAcclMT.h"

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

vector<string> images{"dog.png","strawberry.jpg"};

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
    GTEST_ASSERT_EQ(sent_num_frames_1.load(),recv_num_frames_1.load());
    GTEST_ASSERT_EQ(sent_num_frames_2.load(),recv_num_frames_2.load());
}



TEST(accl_manual_threading_accuracy_test, single_stream_mobilenet){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    MX::Types::MxModelInfo model_info = accl.get_model_info(0);


    fs::path img_path = mx_accl_path/"tests"/"dog.png";
    cv::Mat input_image = mobilenet_load_image(img_path.c_str());
    std::vector<float*> input_data;
    int stream_label = 0;
    input_data.push_back((float*)input_image.data);
    accl.send_input(input_data, model_info.model_index, stream_label,0, false);
    sent_num_frames_1++;

    float* fmap = new float[1000];
    std::vector<float*> ofmap;
    ofmap.reserve(model_info.num_out_featuremaps);
    int recvinf_label=0;
    ofmap.push_back(fmap);
    EXPECT_TRUE(accl.receive_output(ofmap, model_info.model_index, recvinf_label,0, true));
    recv_num_frames_1++;
    vector<float> floatVector(ofmap[0], ofmap[0] + 1000);
    vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);

    EXPECT_EQ(235,top5[0]);
    EXPECT_EQ(0, recvinf_label);
    delete[] fmap;
    test_num_frames();
}


void send(MX::Runtime::MxAcclMT* accl, int num_frames, int stream_idx){
    int i = 0;
    while(++i <= num_frames){
        fs::path img_path = mx_accl_path/"tests"/images[stream_idx];
        cv::Mat input_image = mobilenet_load_image(img_path.c_str());
        std::vector<float*> input_data;
        input_data.push_back((float*)input_image.data);
        accl->send_input(input_data, 0, stream_idx,0, false);
        if(stream_idx==0){
            sent_num_frames_1++;
        }
        else{
            sent_num_frames_2++;
        }
    }
}

void receive(MX::Runtime::MxAcclMT* accl, int num_frames, int stream_idx, int32_t timeout){
    int i = 0;
    while(++i<=num_frames){
        float* fmap = new float[1000];
        std::vector<float*> ofmap;
        ofmap.reserve(1);
        ofmap.push_back(fmap);
        if(!accl->receive_output(ofmap, 0, stream_idx,0, false,timeout)){
            delete[] fmap;
            break;
        }
        
        vector<float> floatVector(ofmap[0], ofmap[0] + 1000);
        vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);
        if(stream_idx == 0){
            EXPECT_EQ(235,top5[0]);
            recv_num_frames_1++;
        }
        else{
            EXPECT_EQ(949,top5[0]);
            recv_num_frames_2++;
        }
        delete[] fmap;
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_mobilenet){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    std::thread send_thread = std::thread(send, accl, 20,0);
    std::thread recv_thread = std::thread(receive, accl, 20,0,0);
    std::thread send_thread_1 = std::thread(send, accl, 20,1);
    std::thread recv_thread_1 = std::thread(receive, accl, 20,1,0);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    if(accl != NULL){
        delete accl;
        accl  = NULL;
    }
}


TEST(accl_manual_threading_2_chip_accuracy_test, 2chip_multistream_mobilenet){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet2chip.dfp";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    std::thread send_thread = std::thread(send, std::ref(accl), 20,0);
    std::thread recv_thread = std::thread(receive, std::ref(accl), 20,0,0);
    std::thread send_thread_1 = std::thread(send, std::ref(accl), 20,1);
    std::thread recv_thread_1 = std::thread(receive, std::ref(accl), 20,1,0);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    if(accl != NULL){
        delete accl;
        accl  = NULL;
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_mobilenet_timeout){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    std::thread send_thread = std::thread(send, std::ref(accl), 20,0);
    std::thread recv_thread = std::thread(receive, std::ref(accl), 20,0,1000);
    std::thread send_thread_1 = std::thread(send, std::ref(accl), 20,1);
    std::thread recv_thread_1 = std::thread(receive, std::ref(accl), 20,1,1000);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    if(accl != NULL){
        delete accl;
        accl  = NULL;
    }
}

void send_run(MX::Runtime::MxAcclMT* accl, int num_frames, int stream_idx){
    int i = 0;
    while(++i <= num_frames){
        fs::path img_path = mx_accl_path/"tests"/images[stream_idx];
        cv::Mat input_image = mobilenet_load_image(img_path.c_str());
        std::vector<float*> input_data;
        input_data.push_back((float*)input_image.data);
        float* fmap = new float[1000];
        std::vector<float*> ofmap;
        ofmap.reserve(1);
        ofmap.push_back(fmap);
        if(!accl->run(input_data,ofmap,0,stream_idx)){
            delete[] fmap;
            FAIL();
            break;
        }
        vector<float> floatVector(ofmap[0], ofmap[0] + 1000);
        vector<size_t> top5 = getTopNMaxIndices(floatVector, 5);
        if(stream_idx==0){
            EXPECT_EQ(235,top5[0]);
        }
        else{
            EXPECT_EQ(949,top5[0]);
        }
        delete[] fmap;
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_mobilenet_run){
    init_num_frames();
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    std::thread run_thread = std::thread(send_run, std::ref(accl), 20,0);
    std::thread run_thread_1 = std::thread(send_run, std::ref(accl), 20,1);

    if(run_thread.joinable() && run_thread_1.joinable()){
        run_thread.join();
        run_thread_1.join();
    }
    if(accl != NULL){
        delete accl;
        accl  = NULL;
    }
}
