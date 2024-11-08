#include <gtest/gtest.h>
#include <memx/accl/utils/path.h>
#include <memx/accl/MxAcclMT.h>
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
}

void send(MX::Runtime::MxAcclMT* accl, int streamidx){

    std::vector<float> input;
    int i = 0;
    while(++i <= 20){
        if(streamidx ==0){
            input = {1,2,4,5,2,3,1,3,4,2,1,6};
            sent_num_frames_1++;
        }
        else{
            input = {5,1,7,2,4,3,5,1,2,6,0,2};
            sent_num_frames_2++;
        }
        std::vector<float*> input_data;
        input_data.push_back(input.data());
        accl->send_input(input_data, 0, streamidx, false);
    }
}

void receive(MX::Runtime::MxAcclMT* accl,int stream_idx){
    int i = 0;
    while(++i <= 20){
        float* fmap = new float[2];
        std::vector<float*> ofmap;
        ofmap.push_back(fmap);
        if(!accl->receive_output(ofmap, 0, stream_idx, false)){
            delete[] fmap;
            break;
        }        
        if(stream_idx == 0){

            EXPECT_EQ(170.0,ofmap[0][0]);
            EXPECT_EQ(212.0,ofmap[0][1]);
            recv_num_frames_1++;
        }
        else{
            EXPECT_EQ(188.0,ofmap[0][0]);
            EXPECT_EQ(184.0,ofmap[0][1]);
            recv_num_frames_2++;
        }
        delete[] fmap;
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_onnx){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_onnx.dfp";
    fs::path pre_path = onnx_plugin_path/"prepost_pre.onnx";
    fs::path post_path = onnx_plugin_path/"prepost_post.onnx";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    accl->connect_pre_model(pre_path);
    accl->connect_post_model(post_path);

    std::thread send_thread = std::thread(send, accl,0);
    std::thread recv_thread = std::thread(receive,accl,0);

    std::thread send_thread_1 = std::thread(send, accl, 1);
    std::thread recv_thread_1 = std::thread(receive, accl,1);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    delete accl;
}

void tf_send(MX::Runtime::MxAcclMT* accl, int streamidx){

    int i =0;
    std::vector<float> input;
    while(++i <= 20){
        if(streamidx ==0){
            input = {1,1,2,3,4,4,5,2,2,1,3,6};
            sent_num_frames_1++;
        }
        else{
            input = {5,5,1,1,7,2,2,6,4,0,3,2};
            sent_num_frames_2++;
        }
        std::vector<float*> input_data;
        input_data.push_back(input.data());
        accl->send_input(input_data, 0, streamidx, false);
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_tf){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tf.dfp";
    fs::path pre_path = tf_plugin_path/"prepost_pre.pb";
    fs::path post_path = tf_plugin_path/"prepost_post.pb";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    accl->connect_pre_model(pre_path);
    accl->connect_post_model(post_path);
    std::thread send_thread = std::thread(tf_send, accl,0);
    std::thread recv_thread = std::thread(receive,accl,0);

    std::thread send_thread_1 = std::thread(tf_send, accl, 1);
    std::thread recv_thread_1 = std::thread(receive, accl,1);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    delete accl;
}

void tflite_receive(MX::Runtime::MxAcclMT* accl, int streamidx){
    int i = 0;
    while(++i <= 20){
        float* fmap = new float[2];
        std::vector<float*> ofmap;
        int recvinf_label;
        ofmap.push_back(fmap);
        if(!accl->receive_output(ofmap, 0, streamidx, false)){
            delete[] fmap;
            break;
        }
        
        if(streamidx == 0){

            EXPECT_EQ(95.0,ofmap[0][0]);
            EXPECT_EQ(116.0,ofmap[0][1]);
            recv_num_frames_1++;
        }
        else{
            EXPECT_EQ(104.0,ofmap[0][0]);
            EXPECT_EQ(102.0,ofmap[0][1]);
            recv_num_frames_2++;
        }
        delete[] fmap;
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_tflite){
    init_num_frames();
    fs::path model_path = dfp_path/"prepost_tflite.dfp";
    fs::path pre_path = tflite_plugin_path/"prepost_pre.tflite";
    fs::path post_path = tflite_plugin_path/"prepost_post.tflite";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    accl->connect_pre_model(pre_path);
    accl->connect_post_model(post_path);
    std::thread send_thread = std::thread(tf_send, accl,0);
    std::thread recv_thread = std::thread(tflite_receive,accl,0);

    std::thread send_thread_1 = std::thread(tf_send, accl, 1);
    std::thread recv_thread_1 = std::thread(tflite_receive, accl,1);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    delete accl;
}

void onnx_name_send(MX::Runtime::MxAcclMT* accl, int streamidx){

    std::vector<float> input;
    int i = 0;
    while(++i <= 20){
        if(streamidx ==0){
            input = {1,2,4,5,2,3,1,3,4,2,1,6};
            sent_num_frames_1++;
        }
        else{
            input = {5,1,7,2,4,3,5,1,2,6,0,2};
            sent_num_frames_2++;
        }
        std::vector<float*> input_data;
        input_data.push_back(input.data());
        accl->send_input(input_data, 0, streamidx, 0,true);
    }
}

void onnx_name_receive(MX::Runtime::MxAcclMT* accl,int stream_idx){
    int i = 0;
    while(++i <= 20){
        float* fmap = new float[2];
        float* fmap1 = new float[2];
        std::vector<float*> ofmap;
        ofmap.push_back(fmap);
        ofmap.push_back(fmap1);
        if(!accl->receive_output(ofmap, 0, stream_idx, false)){
            delete[] fmap;
            delete[] fmap1;
            break;
        }        
        if(stream_idx == 0){

            EXPECT_EQ(90.0,ofmap[0][0]);
            EXPECT_EQ(132.0,ofmap[0][1]);
            EXPECT_EQ(45.0,ofmap[1][0]);
            EXPECT_EQ(66.0,ofmap[1][1]);
            recv_num_frames_1++;
        }
        else{
            EXPECT_EQ(108.0,ofmap[0][0]);
            EXPECT_EQ(104.0,ofmap[0][1]);
            EXPECT_EQ(54.0,ofmap[1][0]);
            EXPECT_EQ(52.0,ofmap[1][1]);
            recv_num_frames_2++;
        }
        delete[] fmap;
        delete[] fmap1;
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_named_onnx){
    init_num_frames();
    fs::path model_path = dfp_path/"name_matching_onnx.dfp";
    fs::path post_path = onnx_plugin_path/"name_matching_post.onnx";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    accl->connect_post_model(post_path);

    std::thread send_thread = std::thread(onnx_name_send, accl,0);
    std::thread recv_thread = std::thread(onnx_name_receive,accl,0);

    std::thread send_thread_1 = std::thread(onnx_name_send, accl, 1);
    std::thread recv_thread_1 = std::thread(onnx_name_receive, accl,1);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    delete accl;
}

void tflite_name_send(MX::Runtime::MxAcclMT* accl, int streamidx){

    std::vector<float> input;
    int i = 0;
    while(++i <= 20){
        if(streamidx ==0){
            input = {1,1,2,3,4,4,5,2,2,1,3,6};
            sent_num_frames_1++;
        }
        else{
            input = {5,5,1,1,7,2,2,6,4,0,3,2};
            sent_num_frames_2++;
        }
        std::vector<float*> input_data;
        input_data.push_back(input.data());
        accl->send_input(input_data, 0, streamidx);
    }
}

void tflite_name_receive(MX::Runtime::MxAcclMT* accl,int stream_idx){
    int i = 0;
    while(++i <= 20){
        float* fmap = new float[2];
        float* fmap1 = new float[2];
        std::vector<float*> ofmap;
        ofmap.push_back(fmap);
        ofmap.push_back(fmap1);
        if(!accl->receive_output(ofmap, 0, stream_idx, false)){
            delete[] fmap;
            delete[] fmap1;
            break;
        }        
        if(stream_idx == 0){

            EXPECT_EQ(58.0,ofmap[0][0]);
            EXPECT_EQ(72.0,ofmap[0][1]);
            EXPECT_EQ(45.0,ofmap[1][0]);
            EXPECT_EQ(66.0,ofmap[1][1]);
            recv_num_frames_1++;
        }
        else{
            EXPECT_EQ(85.0,ofmap[0][0]);
            EXPECT_EQ(70.0,ofmap[0][1]);
            EXPECT_EQ(54.0,ofmap[1][0]);
            EXPECT_EQ(52.0,ofmap[1][1]);
            recv_num_frames_2++;
        }
        delete[] fmap;
        delete[] fmap1;
    }
}

TEST(accl_manual_threading_accuracy_test, multistream_named_tflite){
    init_num_frames();
    fs::path model_path = dfp_path/"name_matching_tflite.dfp";
    fs::path post_path = tflite_plugin_path/"name_matching_post.tflite";
    MX::Runtime::MxAcclMT* accl;
    accl = new MX::Runtime::MxAcclMT;
    accl->connect_dfp(model_path);
    accl->connect_post_model(post_path);

    std::thread send_thread = std::thread(tflite_name_send, accl,0);
    std::thread recv_thread = std::thread(tflite_name_receive,accl,0);

    std::thread send_thread_1 = std::thread(tflite_name_send, accl, 1);
    std::thread recv_thread_1 = std::thread(tflite_name_receive, accl,1);

    if(send_thread.joinable() && send_thread_1.joinable()){
        send_thread.join();
        send_thread_1.join();
    }
    if(recv_thread.joinable() && recv_thread_1.joinable()){
        recv_thread.join();
        recv_thread_1.join();
    }
    test_num_frames();
    delete accl;
}
