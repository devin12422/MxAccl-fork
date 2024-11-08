#ifndef DEVICE_MANAGER
#define DEVICE_MANAGER

#include <string>
#include <stdint.h>
#include <atomic>
#include <iostream>
#include <unordered_map>
#include <memx/memx.h>
#include <memx/accl/dfp.h>
#include <memx/accl/MxModel.h>
#include <memx/accl/utils/path.h>
#include <memx/accl/utils/mxTypes.h>


namespace MX
{
  namespace Runtime
  {
    struct dfp_rt_info{
      std::filesystem::path dfp_filename_path;
      Dfp::DfpObject* dfp;
      int dfp_num_chips;//Number of chips DFP is compiled
      float mxa_gen;//Generation of chip DFP is compiled
      int num_models;//Number of models DFP is compiled
      Dfp::DfpMeta dfp_meta;
      std::vector<int> context_ids_vector;
      bool valid;
      bool is_bytes;
      bool use_multigroup_lb;
    };

    struct device_info{
      int chip_count;
      bool is_device_open;
      int number_of_contexts_attached; // should be less than 32 per device
      std::vector<int> contexts_ids_attached;
      int current_config;
      // int last_context_attached;
    };


    class DeviceManager{
      public:
        DeviceManager();
        bool opendfp(const std::filesystem::path dfp_filename, int dfp_tag);
        bool opendfp_bytes(const uint8_t *b, int dfp_tag);
        bool setup_mxa(int dfp_tag, std::vector<int>& pgroup_ids);
        void attach_dfp_to_device(int dfp_tag);
        void download_dfp_to_device(int dfp_tag);
        void init_mx_models(int dfp_tag, std::vector<ModelBase *>* mxmodel_vector );

        //Getter functions
        int get_dfp_num_chips(int dfp_tag);
        int get_dfp_num_models(int dfp_tag);
        bool get_dfp_validity(int dfp_tag);

        void close_all_devices();
        void cleanup__all_dfps();
        // void cleanup_all_setup_maps();
        static void update_context_tracker_id();
        static int get_context_tracker_id();

        // bool dfp_tag_duplicate_check(int dfp_tag);
        void print_available_devices();
        /*
        // Additional get function disabled for now but might need later
        float get_dfp_mxa_gen();
        int get_connected_devices_count();
        int get_available_device_count();
        Dfp::DfpMeta get_dfp_meta();
        */

      private:

        void get_available_devices();
        void throw_chip_exception(int pdfp_chips, int pdevice_chips, int device_id);
        void throw_mxa_gen_exception(int pdfp_num_chips);
        bool configure_device(int device_id, int device_chip_count, int pdfp_num_chips, float pmx_gen);
        void throw_device_not_available_exception(int pdevice_id);
        bool connect_device(int dfp_tag, int device_id);

        void set_power_mode(int device_id, int num_chips);


        using mxmaptype = std::unordered_map<int, MX::Runtime::dfp_rt_info>;
        mxmaptype dfp_mxa_map;

        using mxa_device_map_type = std::unordered_map< int, MX::Runtime::device_info>;
        mxa_device_map_type available_mxa_device_map;

        int all_devices_count;
        int available_devices;
        int required_devices;
        int number_of_context_per_dfp;
        int group_id_passed;
        std::vector<int> available_devices_id;
        std::vector<int> open_devices;

    };// DeviceManager

 } // namespace Runtime
} // namespace MX

#endif
