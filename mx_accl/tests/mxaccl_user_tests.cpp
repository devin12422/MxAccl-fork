#include <gtest/gtest.h>
#include "memx/accl/utils/path.h"
#include "memx/accl/MxAccl.h"
#include "memx/accl/MxAcclMT.h"

namespace fs = std::filesystem;

fs::path mx_accl_path = MX::Utils::mx_get_accl_dir();
fs::path dfp_path = mx_accl_path/"tests"/"models"/"cascadePlus";

bool input_callback(std::vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){return false;}

bool input_callback_int(std::vector<const MX::Types::FeatureMap<uint8_t>*> dst, int stream_id){return false;}

bool output_callback(std::vector<const MX::Types::FeatureMap<float>*> src, int stream_id){return true;}

TEST(accl_user_tests, default_constructor){
    MX::Runtime::MxAccl accl;
    ASSERT_EQ(0,accl.get_num_models());
    ASSERT_EQ(0,accl.get_num_streams());
}

TEST(accl_manual_user_tests, default_constructor){
    MX::Runtime::MxAcclMT accl;
    ASSERT_EQ(0,accl.get_num_models());
}

TEST(accl_user_tests, vector_connect_dfp){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    std::vector<int> group_list={0};
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path,group_list);
    ASSERT_EQ(1,accl.get_num_models());
}

TEST(accl_user_tests, connect_dfp_emptyvector_exception){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    std::vector<int> group_list;
    try
    {
        MX::Runtime::MxAccl accl;
        accl.connect_dfp(model_path,group_list);
    }
    catch(std::runtime_error const & err)
    {
        ASSERT_EQ(err.what(),std::string("device_ids_to_use parameter cannot be empty"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected device parameter exception";
    }
}

TEST(accl_user_tests, vector_connect_dfp_device_exception){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    try
    {
        MX::Runtime::MxAccl accl;
        accl.connect_dfp(model_path,10000);
    }
    catch(std::runtime_error const & err)
    {
        ASSERT_EQ(err.what(),std::string("Try using the available devices given above"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected Device exception";
    }   
}

TEST(accl_user_tests, destructor_stop){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(input_callback,output_callback,0);
    accl.start();
}

TEST(accl_manual_user_tests, vector_connect_dfp){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    std::vector<int> group_list={0};
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path,group_list);
    ASSERT_EQ(1,accl.get_num_models());
}

TEST(accl_manual_user_tests, connect_dfp_emptyvector_exception){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    std::vector<int> group_list;
    try
    {
        MX::Runtime::MxAcclMT accl;
        accl.connect_dfp(model_path,group_list);
    }
    catch(std::runtime_error const & err)
    {
        ASSERT_EQ(err.what(),std::string("device_ids_to_use parameter cannot be empty"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected device parameter exception";
    }   
}

TEST(accl_manual_user_tests, vector_connect_dfp_device_exception){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    std::vector<int> group_list;
    try
    {
        MX::Runtime::MxAcclMT accl;
        accl.connect_dfp(model_path,10000);
    }
    catch(std::runtime_error const & err)
    {
        ASSERT_EQ(err.what(),std::string("Try using the available devices given above"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected Device exception";
    }   
}

TEST(accl_user_tests, invalid_dfp){
    fs::path model_path = dfp_path/"mobilenet_invalid.dfp";
    std::vector<int> group_list;
    try
    {
        MX::Runtime::MxAccl accl;
        accl.connect_dfp(model_path);
    }
    catch(std::runtime_error const & err)
    {
        ASSERT_EQ(err.what(),std::string("Cannot parse dfp file - Please check given dfp"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected Cannot parse dfp file - Please check given dfp";
    }   
}

TEST(accl_manual_user_tests, invalid_dfp){
    fs::path model_path = dfp_path/"mobilenet_invalid.dfp";
    std::vector<int> group_list;
    try
    {
        MX::Runtime::MxAcclMT accl;
        accl.connect_dfp(model_path);
    }
    catch(std::runtime_error const & err)
    {
        ASSERT_EQ(err.what(),std::string("Cannot parse dfp file - Please check given dfp"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected Cannot parse dfp file - Please check given dfp";
    }   
}

TEST(accl_user_tests, multiple_processes){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    MX::Runtime::MxAccl accl1;
    accl.connect_dfp(model_path);
    try
    {
        accl1.connect_dfp(model_path);
    }
    catch(std::runtime_error const & err)
    {
        std::cout<<err.what()<<std::endl;
        ASSERT_EQ(err.what(),std::string("Cannot acquire lock on an available MXA device, device ID = 0"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected group lock error";
    }        
}

TEST(accl_user_tests, correct_callback_1){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0);
    GTEST_ASSERT_EQ(1,accl.get_num_streams());
}

TEST(accl_user_tests, correct_connect_1){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0);
    accl.start();
    accl.wait(); 
    accl.stop();
    GTEST_ASSERT_EQ(1,accl.get_num_streams());
}

TEST(accl_user_tests, multiple_starts_1){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0);
    accl.start();
    accl.wait();
    accl.stop();
    GTEST_ASSERT_EQ(1,accl.get_num_streams()); 
    accl.connect_stream(&input_callback,&output_callback,1);
    accl.start();
    accl.wait();
    accl.stop();
    GTEST_ASSERT_EQ(2,accl.get_num_streams());    
}

TEST(accl_user_tests, multiple_starts_2){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0);
    accl.start();
    accl.wait();
    accl.stop();
    GTEST_ASSERT_EQ(1,accl.get_num_streams()); 
    accl.start();
    accl.wait();
    accl.stop();
    GTEST_ASSERT_EQ(1,accl.get_num_streams());    
}

TEST(accl_user_tests, multiple_starts_3){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0, 0);
    accl.start();
    accl.wait();
    accl.stop();
    GTEST_ASSERT_EQ(1,accl.get_num_streams()); 
    accl.connect_stream(&input_callback,&output_callback,1, 1);
    accl.start();
    accl.wait();
    accl.stop();
    GTEST_ASSERT_EQ(2,accl.get_num_streams());    
}

TEST(accl_user_tests, invalid_connect_1){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0);
    accl.start();
    try
    {
        accl.connect_stream(&input_callback,&output_callback,1);
    }
    catch(std::logic_error const & err)
    {
        EXPECT_EQ(err.what(),std::string("connect_stream called after starting MxAccl"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected connect_stream error";
    }
    accl.stop();
}

TEST(accl_user_tests, invalid_connect_2){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    try
    {
        accl.connect_stream(&input_callback,&output_callback,0);
        accl.connect_stream(&input_callback,&output_callback,0);
    }
    catch(std::logic_error const & err)
    {
        EXPECT_EQ(err.what(),std::string("duplicate stream id passed in connect_stream"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected duplicate stream id error";
    }
}

TEST(accl_user_tests, invalid_connect_3){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    try
    {
        accl.connect_stream(&input_callback,&output_callback,0);
        accl.connect_stream(&input_callback,&output_callback,0,1);
    }
    catch(std::logic_error const & err)
    {
        EXPECT_EQ(err.what(),std::string("duplicate stream id passed in connect_stream"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected duplicate stream id error";
    }
}

TEST(accl_user_tests, invalid_connect_4){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    try
    {
        accl.start();
    }
    catch(std::logic_error const & err)
    {
        EXPECT_EQ(err.what(),std::string("accl start called before connect_stream for auto threading"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected accl start called before connect_stream for auto threading";
    }
    accl.stop();
}

TEST(accl_user_tests, invalid_wait){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0);
    try
    {
        accl.wait();
    }
    catch(std::logic_error const & err)
    {
        EXPECT_EQ(err.what(),std::string("Model wait called when model is not running"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected model wait error";
    }
}

TEST(accl_user_tests, invalid_stop){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.connect_stream(&input_callback,&output_callback,0);
    try
    {
        accl.stop();
    }
    catch(std::logic_error const & err)
    {
        EXPECT_EQ(err.what(),std::string("Model stop called when model is not running"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected model stop error";
    }
}

// TEST(accl_user_tests, invalid_callback_1){
//     fs::path model_path = dfp_path/"mobilenet.dfp";
//     MX::Runtime::MxAccl accl(model_path);
//     try
//     {
//         accl.connect_stream(&input_callback_int,&output_callback,0);
//     }
//     catch(std::runtime_error const & err)
//     {
//         EXPECT_EQ(err.what(),std::string("base connect_stream int is called"));
//     }
//     catch(...){
//         GTEST_FAIL()<< "Expected callback error";
//     }
// }

TEST(accl_user_tests, invalid_callback_2){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    try
    {
        accl.connect_stream(&input_callback,NULL,0);
    }
    catch(std::invalid_argument const & err)
    {
        EXPECT_EQ(err.what(),std::string("input callback or output callback got a NULL ptr!"));
    }
    catch(...){
        GTEST_FAIL()<< "Expected callback error";
    }
}

int num_frames = 0;
bool input_callback_fmap(std::vector<const MX::Types::FeatureMap<float>*> dst, int stream_id){
    if(num_frames<20){
        EXPECT_EQ(stream_id,dst[0]->get_num_fmap_threads());
        float* in_data = new float[3*224*224];
        dst[0]->set_data(in_data);
        num_frames++;
        delete[] in_data;
    }
    else{
        return false;
    }
    return true;
}

bool output_callback_fmap(std::vector<const MX::Types::FeatureMap<float>*> src, int stream_id){
    EXPECT_EQ(stream_id,src[0]->get_num_fmap_threads());
    return true;
}

TEST(accl_user_tests,num_fmap_threads){
    fs::path model_path = dfp_path/"mobilenet_multimodel.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    accl.set_parallel_fmap_convert(2,0);
    accl.set_parallel_fmap_convert(5,1);
    accl.connect_stream(&input_callback_fmap,&output_callback_fmap,2);
    accl.connect_stream(&input_callback_fmap,&output_callback_fmap,5,1);
    accl.start();
    accl.wait();
    accl.stop();

    const char* expected_exception = "Invalid model ID passed : Number of models available = 2\n model_id range is 0 to 1";
    try
    {
        accl.set_parallel_fmap_convert(2,2);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }
}

TEST(accl_user_tests,set_num_workers_exception){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "Invalid model ID passed : Number of models available = 1\n model_id range is 0 to 0";
    try
    {
        accl.set_num_workers(1,1,1);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }
}

TEST(accl_user_tests,dfp_num_chips){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    ASSERT_EQ(4,accl.get_dfp_num_chips());
}

TEST(accl_manual_user_tests,dfp_num_chips){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    ASSERT_EQ(4,accl.get_dfp_num_chips());
}

TEST(accl_manual_user_tests,num_fmap_threads){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    try
    {
        accl.set_parallel_fmap_convert(2,0);
    }
    catch(const std::exception& e)
    {
        FAIL()<<"set_num_fmaps_threads failed for manual threading";
    }

    const char* expected_exception = "Invalid model ID passed : Number of models available = 1\n model_id range is 0 to 0";
    try
    {
        accl.set_parallel_fmap_convert(1,1);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }  
}

TEST(accl_user_tests,prepost_keras_test){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "The connected model, mobilenet_pre.keras is a keras model which is not supported by MX_API as keras doesn't have a C++ API. So we suggest you to convert the model to Tflite and instructions can be found here, https://www.tensorflow.org/lite/models/convert/convert_models#convert_a_keras_model_";
    try
    {
        accl.connect_pre_model("mobilenet_pre.keras");
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
}

TEST(accl_user_tests,prepost_unknown_test){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "Unknown file extension passed for pre-processing or post-processing";
    try
    {
        accl.connect_pre_model("mobilenet_pre.random");
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
}

TEST(accl_user_tests,invalid_dfp_connect){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "Only one dfp allowed per Accl object";
    try
    {
        accl.connect_dfp(model_path);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
}

TEST(accl_user_tests,invalid_connect_stream){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAccl accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "only one dfp per MxAccl allowed";
    try
    {
        accl.connect_stream(input_callback,output_callback,0,0,1);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
}

TEST(accl_manual_user_tests,invalid_dfp_connect){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "Only one dfp allowed per Accl object";
    try
    {
        accl.connect_dfp(model_path);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
}

TEST(accl_manual_user_tests,invalid_send_input){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "only one dfp per MxAccl allowed";
    std::vector<float*> dummy;
    try
    {
        accl.send_input(dummy,0,0,1);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
    expected_exception = "Invalid model ID passed : Number of models available = 1\n model_id range is 0 to 0";
    try
    {
        accl.send_input(dummy,1,0);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    } 
}

TEST(accl_manual_user_tests,invalid_recv_output){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "only one dfp per MxAccl allowed";
    std::vector<float*> dummy;
    try
    {
        accl.receive_output(dummy,0,0,1);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
    expected_exception = "Invalid model ID passed : Number of models available = 1\n model_id range is 0 to 0";
    try
    {
        accl.receive_output(dummy,1,0);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    } 
}

TEST(accl_manual_user_tests,invalid_run){
    fs::path model_path = dfp_path/"mobilenet.dfp";
    MX::Runtime::MxAcclMT accl;
    accl.connect_dfp(model_path);
    const char* expected_exception = "only one dfp per MxAccl allowed";
    std::vector<float*> dummy;
    try
    {
        accl.run(dummy,dummy,0,0,1);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
    expected_exception = "Invalid model ID passed : Number of models available = 1\n model_id range is 0 to 0";
    try
    {
        accl.run(dummy,dummy,1,0);
    }
    catch(std::exception const & err)
    {
        EXPECT_EQ(std::string(err.what()),expected_exception);
    }
    catch(...){
        FAIL()<< expected_exception;
    }   
}
