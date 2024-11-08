#include <gtest/gtest.h>
#include <thread>
#include "memx/accl/utils/path.h"
#include "memx/accl/MxAccl.h"

using namespace std;
namespace fs = std::filesystem;

fs::path mx_accl_path = MX::Utils::mx_get_accl_dir();
fs::path dfp_path = mx_accl_path/"tests"/"models"/"cascadePlus";


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
    EXPECT_EQ(sent_num_frames_1.load(),recv_num_frames_1.load());
    EXPECT_EQ(sent_num_frames_2.load(),recv_num_frames_2.load());
}

bool input_callback_1(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=20)
        return false;
    float* input = new float[3];
    input[0] = 1;
    input[1] = 2;
    input[2] = 3;
    dst[0]->set_data(input);
    sent_num_frames_1++;
    delete[] input;
    return true;
}

bool output_callback_1(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float;
    src[0]->get_data(output_1);
    EXPECT_EQ(6.0,*output_1);
    recv_num_frames_1++;
    delete output_1;
    return true;
}

bool input_callback_2(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_2.load()>=20)
        return false;
    float* input = new float[3];
    input[0] = 2;
    input[1] = 3;
    input[2] = 4;
    dst[0]->set_data(input);
    sent_num_frames_2++;
    delete[] input;
    return true;
}

bool output_callback_2(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_2 = new float;
    src[0]->get_data(output_2);
    EXPECT_EQ(9.0,*output_2);
    recv_num_frames_2++;
    delete output_2;
    return true;
}

// bool intput_callback_3(vector<const MX::Types::FeatureMap<uint8_t>*> dst, int stream_id){
//     if(sent_num_frames_3.load()>=20)
//         return false;
//     uint8_t* input = new uint8_t[3];
//     input[0] = 3;
//     input[1] = 2;
//     input[2] = 1;
//     dst[0]->set_data(input);
//     sent_num_frames_3++;
//     delete[] input;
//     return true;
// }

bool input_callback_false(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    return false;
}

bool output_callback_false(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    recv_num_frames_1++;
    return true;
}

TEST(accl_dataflow_tests, identity_1){
    init_num_frames();
    fs::path model_path = dfp_path/"identity.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait(); //waiting for a few callbacks to happen
    accl.stop();
    test_num_frames();
}

TEST(accl_dataflow_tests, identity_2){
    init_num_frames();
    fs::path model_path = dfp_path/"identity.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_dataflow_tests, identity_3){
    init_num_frames();
    fs::path model_path = dfp_path/"identity.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_1,&output_callback_1,1);
    accl.connect_stream(&input_callback_2,&output_callback_2,2);
    accl.connect_stream(&input_callback_2,&output_callback_2,3);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_dataflow_tests, identity_4){
    init_num_frames();
    fs::path model_path = dfp_path/"identity_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,1,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

TEST(accl_dataflow_tests, identity_5){
    init_num_frames();
    fs::path model_path = dfp_path/"identity_multimodel.dfp";
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

TEST(accl_dataflow_tests, identity_6){
    init_num_frames();
    fs::path model_path = dfp_path/"identity_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.connect_stream(&input_callback_2,&output_callback_2,1);
    accl.connect_stream(&input_callback_1,&output_callback_1,2,1);
    accl.connect_stream(&input_callback_2,&output_callback_2,3,1);
    accl.connect_stream(&input_callback_1,&output_callback_1,4);
    accl.connect_stream(&input_callback_2,&output_callback_2,5);
    accl.connect_stream(&input_callback_1,&output_callback_1,6,1);
    accl.connect_stream(&input_callback_2,&output_callback_2,7,1);
    accl.connect_stream(&input_callback_1,&output_callback_1,8);
    accl.connect_stream(&input_callback_2,&output_callback_2,9);
    accl.connect_stream(&input_callback_1,&output_callback_1,10,1);
    accl.connect_stream(&input_callback_2,&output_callback_2,11,1);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}

// TEST(accl_dataflow_tests, identity_int){
//     init_num_frames();
//     fs::path model_path = dfp_path/"identity_int.dfp";
//     MX::Runtime::MxAccl accl(model_path.c_str());
//     accl.connect_stream(&intput_callback_3,&output_callback_1,0);
//     accl.start();
//     accl.wait();
//     accl.stop();
// }


bool input_callback_id_flow(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=20)
        return false;
    float* input = new float[3];
    if(stream_id==0 || stream_id==2){
        input[0] = 1;
        input[1] = 2;
        input[2] = 3;
    }
    else if(stream_id==1 || stream_id==3){
        input[0] = 2;
        input[1] = 4;
        input[2] = 5;        
    }
    else{
        ADD_FAILURE();
    }
    dst[0]->set_data(input);
    sent_num_frames_1++;
    delete[] input;
    return true;
}

bool output_callback_id_flow(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float;
    src[0]->get_data(output_1);
    if(stream_id==0 || stream_id==2)
    EXPECT_EQ(6.0,*output_1);
    else if(stream_id==1 || stream_id==3)
    EXPECT_EQ(11.0,*output_1);
    else
    ADD_FAILURE();
    recv_num_frames_1++;
    delete output_1;
    return true;
}

TEST(accl_dataflow_tests, identity_stream_flow_1){
    init_num_frames();
    fs::path model_path = dfp_path/"identity.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,0);
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,1);
    accl.start();
    accl.wait();
    accl.stop();
}

TEST(accl_dataflow_tests, identity_stream_flow_2){
    init_num_frames();
    fs::path model_path = dfp_path/"identity_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,0);
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,1,1);
    accl.start();
    accl.wait();
    accl.stop();
}

TEST(accl_dataflow_tests, identity_stream_flow_3){
    init_num_frames();
    fs::path model_path = dfp_path/"identity_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,0);
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,1);
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,2,1);
    accl.connect_stream(&input_callback_id_flow,&output_callback_id_flow,3,1);
    accl.start();
    accl.wait();
    accl.stop();
}
vector<int> num_frames{0,0,0,0,0,0,0,0,0};
vector<int> recv_num_frames{0,0,0,0,0,0,0,0,0};

bool input_callback_fixed_frames(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(num_frames[stream_id]>=20){ 
        return false;
    }
    // this_thread::sleep_for(std::chrono::milliseconds(30));
    float* input = new float[3];
    if(stream_id%3==0){
        input[0] = 1;
        input[1] = 2;
        input[2] = 3;
    }
    else if(stream_id%3==1){
        input[0] = 2;
        input[1] = 4;
        input[2] = 5;        
    }
    else if(stream_id%3==2){
        input[0] = 7;
        input[1] = 8;
        input[2] = 9;        
    }
    else{
        ADD_FAILURE();
    }
    dst[0]->set_data(input);
    sent_num_frames_1++;
    delete[] input;
    num_frames[stream_id]++;
    return true;
}

bool output_callback_fixed_frames(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float;
    src[0]->get_data(output_1);
    if(stream_id%3==0)
    EXPECT_EQ(6.0,*output_1);
    else if(stream_id%3==1)
    EXPECT_EQ(11.0,*output_1);
    else if (stream_id%3==2)
    EXPECT_EQ(24.0,*output_1);
    else
    ADD_FAILURE();
    recv_num_frames[stream_id]++;
    delete output_1;
    return true;
}

TEST(accl_dataflow_tests, identity_fixed_num_frames){
    // init_num_frames();
    fs::path model_path = dfp_path/"identity_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,0);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,1);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,2,1);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,3,1);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,4);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,5);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,6,1);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,7,1);
    accl.connect_stream(&input_callback_fixed_frames,&output_callback_fixed_frames,8,1);
    accl.start();
    accl.wait();//waiting for 1000 frames
    accl.stop();
    for(int i=0; i<9;++i){
        ASSERT_EQ(num_frames[i],20);
        ASSERT_EQ(recv_num_frames[i],20);
    }
}


TEST(accl_dataflow_tests, identity_false){
    init_num_frames();
    fs::path model_path = dfp_path/"identity.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_false,&output_callback_false,0);
    accl.start();
    accl.wait();
    accl.stop();
    EXPECT_EQ(recv_num_frames_1,0);
}


TEST(accl_dataflow_tests, identity_multiconfig){
    init_num_frames();
    fs::path model_path = dfp_path/"identity_2x2.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_1,&output_callback_1,0);
    accl.start();
    accl.wait();
    accl.stop();
}

bool input_callback_gbf(vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(sent_num_frames_1.load()>=20)
        return false;
    float* input = new float[6];
    input[0] = 1;
    input[1] = 2;
    input[2] = 3;
    input[3] = 4;
    input[4] = 5;
    input[5] = 6;
    dst[0]->set_data(input);
    sent_num_frames_1++;
    delete[] input;
    return true;
}

bool output_callback_gbf(vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    float* output_1 = new float[5];
    float* output_2 = new float[5];
    src[0]->get_data(output_1);
    src[1]->get_data(output_2);
    EXPECT_EQ(5.0,output_1[0]);
    EXPECT_EQ(8.0,output_1[1]);
    EXPECT_EQ(11.0,output_1[2]);
    EXPECT_EQ(14.0,output_1[3]);
    EXPECT_EQ(17.0,output_1[4]);

    EXPECT_EQ(10.0,output_2[0]);
    EXPECT_EQ(16.0,output_2[1]);
    EXPECT_EQ(22.0,output_2[2]);
    EXPECT_EQ(28.0,output_2[3]);
    EXPECT_EQ(34.0,output_2[4]);

    recv_num_frames_1++;
    delete[] output_1;
    delete[] output_2;
    return true;
}

TEST(accl_dataflow_tests, gbf_identity){
    init_num_frames();
    fs::path model_path = dfp_path/"gbfmodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path.c_str());
    accl.connect_stream(&input_callback_gbf,&output_callback_gbf,0);
    accl.start();
    accl.wait();
    accl.stop();
    test_num_frames();
}
