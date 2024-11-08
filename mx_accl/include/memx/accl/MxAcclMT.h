#ifndef MX_ACCLMT
#define MX_ACCLMT

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
    class MxAcclMT{

      public:

      /**
       * @brief MxAcclMT constructor. Takes no arguments.
       */
      MxAcclMT();

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
       * @brief Construct a new MxAccl object. Currently only one connect_dfp per MxAccl object is allowed.
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
      ~MxAcclMT();

      /**
       * @brief Get number of models in the compiled DFP
       *
       * @return Number of models
       */
      int get_num_models();

      /**
       * @brief Get number of chips the dfp is compiled for
       *
       * @return Number of chips
       */
      int get_dfp_num_chips();


      /**
       * @brief get information of a particular model such as number of in out featureMaps and in out layer names
       * @param model_id model ID or the index for the required information
       * @return if valid model_id then MxModelInfo model_info with necessary information else throw runtime error invalid model_id
      */
      MX::Types::MxModelInfo get_model_info(int model_id) const;

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

      /**
       * @brief Send input to the accelerator in userThreading mode.
       *
       * @param in_data -> vector of input data to the model
       * @param model_id -> Index of the model the data is targetted to.
       * @param stream_id -> Index of stream the input data belongs to.
       * @param dfp_id -> id of dfp returned by connect_dfp() function
       * @param channel_first -> boolean variable that indicates the copied data is in channel first or channle last format. default is false expecting data in channel last format
       * @param timeout -> Wait time in milliseconds for the function to be succesful. Default is 0 which indicates that the function never timesout.
       * @return Returns true if the inference is succesful and false if a timeout happens.
      */
      bool send_input(std::vector<float*> in_data, int model_id, int stream_id, int dfp_id=0, bool channel_first = false, int32_t timeout = 0);


      // /**
      //  * @brief Send input to the accelerator in userThreading mode.
      //  *
      //  * @param in_data -> vector of input data to the model
      //  * @param model_id -> Index of the model the data is targetted to.
      //  * @param stream_id -> Index of stream the input data belongs to.
      //  * @param channel_first -> boolean variable that indicates the copied data is in channel first or channle last format. default is false expecting data in channel last format
      //  * @param timeout -> Wait time in milliseconds for the function to be succesful. Default is 0 which indicates that the function never timesout.
      //  * @return Returns true if the inference is succesful and false if a timeout happens.
      // */
      // bool send_input(std::vector<uint8_t*> in_data, int model_id, int stream_id, bool channel_first = false, int32_t timeout = 0);

      /**
       * @brief Receive output from the accelerator in userThreading mode.
       *
       * @param out_data -> vector of output data from the model
       * @param model_id -> Index of the model the data is intended to come from.
       * @param stream_id -> Index of stream the output data belongs to.
       * @param dfp_id -> id of dfp returned by connect_dfp() function
       * @param channel_first -> boolean variable that indicates the copied data is in channel first or channle last format. default is false expecting data in channel last format
       * @param timeout -> Wait time in milliseconds for the function to be succesful. Default is 0 which indicates that the function never timesout.
       * @return Returns true if the inference is succesful and false if a timeout happens.
      */
      bool receive_output(std::vector<float*> &out_data, int model_id, int stream_id, int dfp_id=0, bool channel_first = false, int32_t timeout=0);

      // /**
      //  * @brief Run inference on the accelerator in userThreading mode.
      //  *
      //  * @param in_data -> vector of input data to the model
      //  * @param out_data -> vector of output data from the model
      //  * @param model_id -> Index of the model the data is intended to come from.
      //  * @param stream_id -> Index of stream the output data belongs to.
      //  * @param dfp_id -> id of dfp returned by connect_dfp() function
      //  * @param in_channel_first -> boolean variable that indicates the copied input data is in channel first or channle last format. default is false expecting data in channel last format
      //  * @param in_channel_first -> boolean variable that indicates the copied output data is in channel first or channle last format. default is false expecting data in channel last format
      //  * @param timeout -> Wait time in milliseconds for the function to be succesful. Default is 0 which indicates that the function never timesout.
      //  * @return Returns true if the inference is succesful and false if a timeout happens.
      // */
      // bool run(std::vector<uint8_t *> in_data, std::vector<float*> &out_data, int pmodel_id, int pstream_id, int dfp_id=0, bool in_channel_first=false, bool out_channel_first=false, int32_t timeout=0);

      /**
       * @brief Run inference on the accelerator in userThreading mode.
       *
       * @param in_data -> vector of input data to the model
       * @param out_data -> vector of output data from the model
       * @param model_id -> Index of the model the data is intended to come from.
       * @param stream_id -> Index of stream the output data belongs to.
       * @param dfp_id -> id of dfp returned by connect_dfp() function
       * @param in_channel_first -> boolean variable that indicates the copied input data is in channel first or channle last format. default is false expecting data in channel last format
       * @param in_channel_first -> boolean variable that indicates the copied output data is in channel first or channle last format. default is false expecting data in channel last format
       * @param timeout -> Wait time in milliseconds for the function to be succesful. Default is 0 which indicates that the function never timesout.
       * @return Returns true if the inference is succesful and false if a timeout happens.
      */
      bool run(std::vector<float *> in_data, std::vector<float*> &out_data, int pmodel_id, int pstream_id, int dfp_id=0, bool in_channel_first=false, bool out_channel_first=false, int32_t timeout=0);

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
          int dfp_tag;
          bool dfp_valid;
          bool setup_status;

          int group; //Group of the chip connected

          std::atomic_bool manual_run; // Flag to mark manual threading option;

          std::vector<ModelBase *> models;//Vector of all model objects

          MX::Runtime::DeviceManager *device_manager;


    }; // MxAcclMT
  } // namespace Runtime
} // namespace MX

#endif
