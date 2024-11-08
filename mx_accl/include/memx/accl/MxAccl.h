#ifndef MX_ACCL
#define MX_ACCL

#include <string>
#include <stdint.h>
#include <atomic>
#include <thread>

#include <memx/accl/MxModel.h>
#include <memx/accl/dfp.h>
#include <memx/accl/utils/general.h>
#include <memx/accl/utils/featureMap.h>
#include <memx/accl/utils/path.h>
#include <memx/accl/DeviceManager.h>

using namespace std;

namespace MX
{
  namespace Runtime
  {
    class MxAccl
    {
    private:
      typedef std::function<bool(vector<const MX::Types::FeatureMap<uint8_t> *>, int stream_id)> int_callback_t;
      typedef std::function<bool(vector<const MX::Types::FeatureMap<float> *>, int stream_id)> float_callback_t;
    public:

      /**
       * @brief MxAccl constructor. Takes no arguments.
       */
      MxAccl();

      /**
       * @brief Connect a dfp to MxAccl object. Currently only one connect_dfp per MxAccl object is allowed.
       *
       * @param file_path Absolute path of DFP file. char* and String types can also be passed.
       * @param device_ids_to_use IDs of MXA devices this process intends to use. takes in a vector of IDs and will return an error if an empty vector is passed
       *
       * @return dfp_id which is later to be passed in connect_stream function to specify that specific stream to a dfp
       */
      int connect_dfp(const std::filesystem::path dfp_path,std::vector<int>& device_ids_to_use);

      /**
       * @brief Connect a dfp to MxAccl object. Currently only one connect_dfp per MxAccl object is allowed.
       *
       * @param file_path Absolute path of DFP file. char* and String types can also be passed.
       * @param group_id GroupId of MPU this application is intended to use.
       * group_id is defaulted to 0, but needs to be provided if using
       * any other group
       *
       * @return dfp_id which is later to be passed in connect_stream function to specify that specific stream to a dfp
       */
      int connect_dfp(const std::filesystem::path dfp_path,int group_id = 0);

      /**
       * @brief Connect a dfp as bytes to MxAccl object. Currently only one connect_dfp per MxAccl object is allowed.
       *
       * @param dfp_bytes Raw uint8_t* pointer to DFP data
       * @param device_ids_to_use IDs of MXA devices this process intends to use. takes in a vector of IDs and will return an error if an empty vector is passed
       *
       * @return dfp_id which is later to be passed in connect_stream function to specify that specific stream to a dfp
       */
      int connect_dfp(const uint8_t *dfp_bytes, std::vector<int>& device_ids_to_use);

      /**
       * @brief Connect a dfp as bytes to MxAccl object. Currently only one connect_dfp per MxAccl object is allowed.
       *
       * @param dfp_bytes Raw uint8_t* pointer to DFP data
       * @param group_id GroupId of MPU this application is intended to use.
       * group_id is defaulted to 0, but needs to be provided if using
       * any other group
       *
       * @return dfp_id which is later to be passed in connect_stream function to specify that specific stream to a dfp
       */
      int connect_dfp(const uint8_t *dfp_bytes, int group_id = 0);

      //Destructor
      ~MxAccl();

      /**
       * @brief Start running inference.
       * All streams must be connected before calling this function
       */
      void start();

      /**
       * @brief Stop running inference.
       * Shouldn't be called before calling start.
       *
       */
      void stop();

      /**
       * @brief Wait for all the streams to be done streaming. This function waits
       * till all the started input callbacks have returned false.
       * Shouldn't be called before calling start.
       *
       */
      void wait();

      /**
       * @brief Get number of models in the compiled DFP
       *
       * @return Number of models
       */
      int get_num_models();

      /**
       * @brief Get number the number of streams connected to the object
       *
       * @return Number of streams
       */
      int get_num_streams();

      /**
       * @brief Get number of chips the dfp is compiled for
       *
       * @return Number of chips
       */
      int get_dfp_num_chips();

      /**
       * @brief Connect a stream to a model
       * - float_callback_t is a function pointer of type, bool foo(vector<const MX::Types::FeatureMap<float>*>, int).
       * - When this input callback function returns false, the corresponding stream is stopped and when all the streams stop,
       * wait() is executed.
       * - connect_stream should be called before calling start() or after calling stop().
       * @param in_cb -> input callback function used by this stream
       * @param out_cb -> output callback function used by this stream
       * @param stream_id -> Unique id given to this stream which can later
       *              be used in the corresponding callback functions
       * @param model_id -> Index of model this stream is intended to be connected
       * @param dfp_id -> id of dfp returned by connect_dfp() function
      */
      void connect_stream(float_callback_t in_cb, float_callback_t out_cb, int stream_id, int model_id=0, int dfp_id = 0);
      // /**
      //  * @brief Connect a stream to a model
      //  * - float_callback_t is a function pointer of type, bool foo(vector<const MX::Types::FeatureMap<float>*>, int).
      //  * - int_callback_t is a function pointer of type, bool foo(vector<const MX::Types::FeatureMap<int>*>, int).
      //  * - When this input callback function returns false, the corresponding stream is stopped and when all the streams stop,
      //  * wait() is executed.
      //  * - connect_stream should be called before calling start() or after calling stop().
      //  * @param in_cb -> input callback function used by this stream
      //  * @param out_cb -> output callback function used by this stream
      //  * @param stream_id -> Unique id given to this stream which can later
      //  *              be used in the corresponding callback functions
      //  * @param model_id -> Index of model this stream is intended to be connected
      // */
      // void connect_stream(int_callback_t in_cb, float_callback_t out_cb, int stream_id, int model_id=0);

      /**
       * @brief get information of a particular model such as number of in out featureMaps and in out layer names
       * @param model_id model ID or the index for the required information
       * @return if valid model_id then MxModelInfo model_info with necessary information else throw runtime error invalid model_id
      */
      MX::Types::MxModelInfo get_model_info(int model_id) const;

      /**
       * @brief get information of the pre-processing model set to a particular model such as number of in out featureMaps and their sizes and shapes
       * @param model_id model ID or the index for the required information
       * @return if valid model_id then MxModelInfo model_info with necessary information else throw runtime error invalid model_id
      */
      MX::Types::MxModelInfo get_pre_model_info(int model_id) const;

      /**
       * @brief get information of the post-processing model set to a particular model such as number of in out featureMaps and their sizes and shapes s
       * @param model_id model ID or the index for the required information
       * @return if valid model_id then MxModelInfo model_info with necessary information else throw runtime error invalid model_id
      */
      MX::Types::MxModelInfo get_post_model_info(int model_id) const;

      // User threading functions - No doxygen comments as we are releasing this for internal use
      /**
       * @brief Set the number of workers for input and output streams. The default is the number of streams
       * for both number of input and output streams as that provides the maximum performance. If this method is
       * not called before calling start(), the accl will run in default mode. This method should be called
       * after connecting all the required streams.
       *
       * @param input_num_workers Number of input workers
       * @param output_num_workers Number of output workers
       * @param model_idx Index of model to which the workers are intended to be assigned to. The default is set to 0
      */
      void set_num_workers(int input_num_workers, int output_num_workers,int model_idx=0);

      /**
       * @brief Connect the information of the post-processing model that has been cropped by the neural compiler
       *
       * @param post_model_path  Abosulte path of the post-processing model. (Can be onnx/tflite etc)
       * @param model_idx The index of model for which the post-processing is intended to be connected to.
       * @param post_size_list If the output of the post-processing has a variable size or if the ouput sizes
       * are not deduced, the maximum possible sizes of the output need to be passed. The default is an empty vector.
      */
      void connect_post_model(std::filesystem::path post_model_path, int model_idx=0, const std::vector<size_t>& post_size_list={});

      /**
       * @brief Connect the information of the pre-processing model that has been cropped by the neural compiler
       *
       * @param pre_model_path  Abosulte path of the pre-processing model. (Can be onnx/tflite etc)
       * @param model_idx The index of model for which the post-processing is intended to be connected to.
      */
      void connect_pre_model(std::filesystem::path pre_model_path, int model_idx=0);

      /**
       * @brief Configure multi-threaded FeatureMap data conversion using the given number of threads.
       * Conversion multithreading is mainly intended for high FPS single-stream scenarios, or userThreading mode.
       * In multi-stream autoThreading scenarios, this option should not be necessary, and may even
       * degrade performance due to increased CPU load.
       *
       * @param num_threads Number of worker threads for FeatureMaps. Use >= 2 to enable. Values < 2 disable.
       * @param model_idx Index of model to enable the feature  The default is set to 0
      */
      void set_parallel_fmap_convert(int num_threads, int model_idx=0);

    private:
      std::filesystem::path dfp_path;
      std::vector<std::filesystem::path> dfp_paths;
      bool dfp_valid;
      bool setup_status;

      int group; //Group of the chip connected

      int dfp_tag;

      std::atomic_bool run;//Flag to know status of the Accl

      std::vector<ModelBase *> models;//Vector of all model objects

      MX::Runtime::DeviceManager *device_manager;

    };
  } // namespace Runtime
} // namespace MX

#endif
