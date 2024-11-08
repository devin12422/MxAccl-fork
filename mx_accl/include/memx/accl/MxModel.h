#ifndef MX_MODEL
#define MX_MODEL

#include <iostream>
#include <vector>
#include <stdint.h>
#include <cstdio>
#include <thread>
#include <functional>
#include <atomic>
#include <cstring>
#include <unordered_set>
#include <filesystem>

#include <memx/memx.h>
#include <memx/accl/dfp.h>
#include <memx/accl/prepost.h>
#include <memx/accl/utils/general.h>
#include <memx/accl/utils/thread_pool.hpp>
#include <memx/accl/utils/featureMap.h>
#include <memx/accl/utils/errors.h>
#include <memx/accl/utils/mxTypes.h>

using namespace std;

namespace MX
{
    namespace Runtime
    {
        /**
         * Base Model class that needs to be inherited by template Model class
         * This class provides virtual funtions required by user that are
         * supposed to be overriden by child classes
        */
        class ModelBase{
            public:
            //function refernce for callback functions
            typedef std::function<bool(vector<const MX::Types::FeatureMap<uint8_t> *>, int)> int_callback_t;
            typedef std::function<bool(vector<const MX::Types::FeatureMap<float> *>, int)> float_callback_t;

            //connect_stream to this Model
            virtual void connect_stream(float_callback_t, float_callback_t, int)
                                            {
                                                throw runtime_error("base connect_stream float is called");
                                            }

            //connect_stream to this Model
            virtual void connect_stream(int_callback_t, float_callback_t, int)
                                            {
                                                throw runtime_error("base connect_stream int is called");
                                            }

            //Set number of workers
            virtual void set_num_workers(int, int)=0;
            // Set multi-thread FMap conversion threads
            virtual void set_parallel_fmap_convert(int)=0;
            //Start the model
            virtual void model_start()=0;

            //Stop the model
            virtual void model_stop()=0;

            // manual threading model start function to init model and featureMaps
            virtual void model_manual_start()=0;

            // // manual threading model stop function to init model and featureMaps
            virtual void model_manual_stop()=0;

            // manual threading model send for float
            virtual bool model_manual_send(std::vector<float*>, int, bool, int32_t ){
                throw runtime_error("base model manual send for float is called");
            };

            // manual threading send for int
            // virtual bool model_manual_send(std::vector<uint8_t*> , int , bool , int32_t ){
            //     throw runtime_error("base model manual send for int is called");
            // };

            // manual threadin send for float
            virtual bool model_manual_receive(std::vector<float*> &, int, bool, int32_t)=0;
            //Get num streams in this model
            virtual int get_num_streams()=0;

            //Wait for model to finish
            virtual void model_wait()=0;

            virtual MX::Types::MxModelInfo return_model_info()=0;

            virtual MX::Types::MxModelInfo return_pre_model_info()=0;

            virtual MX::Types::MxModelInfo return_post_model_info()=0;

            virtual void model_set_post(std::filesystem::path post_model_path, const std::vector<size_t>&)=0;

            virtual void model_set_pre(std::filesystem::path pre_model_path)=0;

            // virtual bool manual_run(std::vector<uint8_t *>, std::vector<float*> &, int , bool, bool, int32_t){return false;};

            virtual bool manual_run(std::vector<float *>, std::vector<float*> &, int , bool, bool, int32_t){return false;};

            // virtual void log_model_info(){throw runtime_error("base print info is called");};

            virtual ~ModelBase(){};
        };

        template <typename T>
        class MxModel : public ModelBase
        {
        private:
            int model_id_; // unique id of the model on MXA
            int group_id_; // unique id of MXA
            int num_streams_;// num streams connected to this model
            Dfp::DfpObject *dfp_; // Dfp object
            MX::Utils::fifo_queue<int> stream_queue; //queue to store stream ids for ifmaps

            vector<uint8_t> in_ports_; // input port information
            vector<uint8_t> out_ports_; // output port information
            int input_num_workers_;
            int output_num_workers_;
            thread_pool* input_pool;
            thread_pool* output_pool;
            void model_send_fun(); //send thread function to perform ifmap
            void model_recv_fun(); //recv thread function to perform ofmap
            thread *model_send_thread; //send thread for model
            thread *model_recv_thread; //recv thread for model
            thread *model_manual_recv_thread; //recv thread for model
            void model_manual_recv_fun(); //recv thread function to perform ofmap
            atomic_bool model_run; //flag to specify if model is running
            atomic_bool model_recv_run;// flag to specify if model recv thread is running
            atomic_bool model_manual_run; //flag to specify manual threadin is opted out
            typedef std::function<bool(vector<const MX::Types::FeatureMap<T> *>, int stream_id)> combined_input_callback_t;
            typedef std::function<bool(vector<const MX::Types::FeatureMap<float> *>, int stream_id)> combined_output_callback_t;
            //Task done by each worker of input threadpool
            bool inputTask(combined_input_callback_t in_cb, vector<const MX::Types::FeatureMap<T>* >inputs,int stream, int stream_idx);
            //Task done by each worker of output threadpool
            bool outputTask(combined_output_callback_t out_cb, vector<const MX::Types::FeatureMap<float>* >outputs,int stream, int stream_idx);
            //vector of input callback functions
            vector<combined_input_callback_t> comb_in_call;
            //vector of output callback functions
            vector<combined_output_callback_t> comb_out_call;
            //set of stream ids connected to the whole accl
            unordered_set<int> stream_set_;
            //list of streamids connected to the model
            vector<int> stream_id_list;

            //Vector of featureMaps of size num_streams that holds inputs for pre-processing models
            vector<vector<MX::Types::FeatureMap<T> *>> pre_in_featuremaps_;
            //Vector of featureMaps of size num_streams that holds inputs for models
            vector<vector<MX::Types::FeatureMap<T> *>> in_featuremaps_;
            //Vector of featureMaps of size num_streams that holds outputs for models
            vector<vector<MX::Types::FeatureMap<float> *>> out_featuremaps_;
            //Vector of featureMaps of size num_streams that holds inputs for post-processing models
            vector<vector<MX::Types::FeatureMap<float> *>> post_out_featuremaps_;

            //model information
            MX::Types::MxModelInfo model_info;
            MX::Types::MxModelInfo pre_model_info;
            MX::Types::MxModelInfo post_model_info;

            PrePost* pre_info_model;
            PrePost* post_info_model;
            float mxa_gen;//Generation of chip DFP is compiled

            vector<MX::Types::FeatureMap<T> *> single_input_featuremap_;
            vector<MX::Types::FeatureMap<float> *> single_output_featuremap_;
            std::condition_variable model_manual_cv;
            std::mutex manual_mutex;
            bool model_manual_in_done;

            //input task
            bool input_task_flag;
            std::mutex input_task_mutex;
            std::condition_variable input_task_cv;

            atomic_int input_thread_counter;
            std::mutex input_thread_mutex;
            std::condition_variable input_thread_cv;

            //model_recv
            bool model_recv_flag;
            std::mutex model_recv_mutex;
            std::condition_variable model_recv_cv;

            vector<std::mutex*> out_task_mutex;
            vector<std::condition_variable*> out_task_cv;

            const std::vector<int>* open_contexts;

            int context_send_current_index=0;
            int number_of_contexts;

            //Queue to pass stream id and context id from send to recv functions
            MX::Utils::fifo_queue<std::pair<int, int>> pair_stream_context_queue;


            //Pre-processing model items
            std::filesystem::path post_model_path_;
            std::vector<PrePost*> post_model;
            std::vector<std::vector<MX::Types::FeatureMap<float>*>> transposed_out_featuremaps_;
            std::vector<size_t> post_out_size;

            //Post-processing model items
            std::filesystem::path pre_model_path;
            std::vector<PrePost*> pre_model;
            std::vector<std::vector<MX::Types::FeatureMap<T>*>> transposed_in_featuremaps_;
            std::vector<size_t> pre_out_size;

            void create_and_append_in_fm();
            void create_and_append_out_fm();

            std::unordered_map<int,int> stream_id_map_;
            std::mutex fm_create_mutex;
            std::mutex manual_mutex_in;
            std::mutex manual_mutex_out;
            std::vector<std::mutex*> manual_recv_mutex;
            std::vector<std::condition_variable*> manual_recv_cv;
            std::vector<bool> manual_recv_flag;
            std::vector<std::mutex*> manual_recv_task_mutex;
            std::vector<std::condition_variable*> manual_recv_task_cv;
            std::vector<bool> manual_recv_task_flag;
            std::mutex manual_init_mutex;
            std::condition_variable manual_init_cv;
            void create_append_manual_mem();

            int parallel_fmap_convert_threads;

            Dfp::DfpMeta meta_;

            void _post_inference(int stream);
            void _pre_inference(int stream);
            void _pre_copy(int stream);

        public:
            MxModel(int model_id, Dfp::DfpObject *dfp_object,  const std::vector<int>* popen_contexts = NULL); // Construct model for Inference

            void model_start() override;
            void model_stop() override;
            void model_wait() override;

            void model_manual_start() override;
            void model_manual_stop() override;
            ~MxModel();
            void connect_stream(combined_input_callback_t in_cb, combined_output_callback_t out_cb, int stream_id) override;
            int get_num_streams() override;
            // void log_model_info() override;
            MX::Types::MxModelInfo return_model_info() override;
            MX::Types::MxModelInfo return_pre_model_info() override;
            MX::Types::MxModelInfo return_post_model_info() override;

            bool model_manual_send(std::vector<T*> in_data, int stream_id, bool channel_first=false, int32_t timeout = 0) override;

            bool model_manual_receive(std::vector<float*> &out_data, int stream_id, bool channel_first=false, int32_t timeout = 0) override;

            bool manual_run(std::vector<T *> in_data, std::vector<float*> &out_data, int pstream_id, bool in_channel_first=false, bool out_channel_first=false, int32_t timeout=0) override;

            void model_set_post(std::filesystem::path post_model_path, const std::vector<size_t>& post_out_size_list) override;

            void model_set_pre(std::filesystem::path pre_model_path) override;
            void set_num_workers(int input_workers, int output_workers) override;
            void set_parallel_fmap_convert(int num_threads) override;
        };
    } // namespace Runtime
} // namespace MX

#endif
