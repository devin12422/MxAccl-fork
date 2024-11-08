#include <memx/accl/utils/featureMap.h>
#include <memx/accl/utils/gbf.h>

#include <cstring>
#include <iostream>
#include <fstream>

using namespace MX::Types;
using namespace MX::Utils;

template <typename T>
FeatureMap<T>::FeatureMap(size_t size, MX_data_format format,  uint16_t dim_h, uint16_t dim_w, uint16_t dim_z, size_t num_chan, int fmap_convert_threads)
{
    fmap_data = new T[size];
    featureMap_size = size;
    fmt = format;
    fmap_convert_threads_ = fmap_convert_threads;
    if(fmt == MX_FMT_RGB565 || fmt == MX_FMT_YUV422){
        throw runtime_error("featureMap given a removed format rgb565/yuv422/yuy2");
    }
    else if(std::is_same<T, float>::value){
        if(fmt == MX_FMT_RGB888)
            throw runtime_error("featureMap<float> was given RGB888 format");
    }
    else if(std::is_same<T, uint8_t>::value){
        if(fmt == MX_FMT_GBF80 || fmt == MX_FMT_BF16 || fmt == MX_FMT_FP32 || fmt == MX_FMT_GBF80_ROW)
            throw runtime_error("featureMap<uint8_t> was given a float-type format");
    } else {
        throw runtime_error("featureMap was given an unknown format!");
    }
    this->dim_h = dim_h;
    this->dim_w = dim_w;
    this->dim_z = dim_z;
    num_ch = num_chan;
    if((fmt == MX_FMT_GBF80 || fmt == MX_FMT_GBF80_ROW) && !(num_ch > 0))
        throw runtime_error("featureMap<float> got an invalid # of channels for MX_FMT_GBF80");
    formatted_data = NULL;
    calc_convert_size_and_new();
    out_ready.store(true);
    in_ready.store(true);
    wait_flag = true;
}

template <typename T>
FeatureMap<T>::FeatureMap(T *in_data, size_t size, MX_data_format format,  uint16_t dim_h, uint16_t dim_w, uint16_t dim_z, size_t num_chan, int fmap_convert_threads)
{
    fmap_data = new T[size];
    featureMap_size = size;
    std::memcpy(fmap_data, in_data, featureMap_size);
    fmt = format;
    fmap_convert_threads_ = fmap_convert_threads;
    if(fmt == MX_FMT_RGB565 || fmt == MX_FMT_YUV422){
        throw runtime_error("featureMap given a removed format rgb565/yuv422/yuy2");
    }
    else if(std::is_same<T, float>::value){
        if(fmt == MX_FMT_RGB888)
            throw runtime_error("featureMap<float> was given RGB888 format");
    }
    else if(std::is_same<T, uint8_t>::value){
        if(fmt == MX_FMT_GBF80 || fmt == MX_FMT_BF16 || fmt == MX_FMT_FP32 || fmt == MX_FMT_GBF80_ROW)
            throw runtime_error("featureMap<uint8_t> was given a float-type format");
    } else {
        throw runtime_error("featureMap was given an unknown format!");
    }
    this->dim_h = dim_h;
    this->dim_w = dim_w;
    this->dim_z = dim_z;
    num_ch = num_chan;
    if((fmt == MX_FMT_GBF80 || fmt == MX_FMT_GBF80_ROW) && !(num_ch > 0))
        throw runtime_error("featureMap<float> got an invalid # of channels for MX_FMT_GBF80");
    formatted_data = NULL;
    calc_convert_size_and_new();
    convert_data();
    out_ready.store(true);
    in_ready.store(true);
    wait_flag = true;
}

template <typename T>
FeatureMap<T>::FeatureMap(const FeatureMap& rhs){
    featureMap_size = rhs.featureMap_size;
    fmt = rhs.fmt;
    this->dim_h = rhs.dim_h;
    this->dim_w = rhs.dim_w;
    this->dim_z = rhs.dim_z;
    num_ch = rhs.num_ch;
    fmap_convert_threads_ = rhs.fmap_convert_threads_;
    formatted_featuremap_size = rhs.formatted_featuremap_size;
    fmap_data = new T[featureMap_size];
    std::memcpy(fmap_data, rhs.fmap_data, featureMap_size);
    if(fmt == MX_FMT_RGB888 || fmt == MX_FMT_FP32){
        formatted_data = (uint8_t*) fmap_data;
    } else {
        formatted_data = new uint8_t[formatted_featuremap_size];
        std::memcpy(formatted_data, rhs.formatted_data, formatted_featuremap_size);
    }
    out_ready.store(true);
    in_ready.store(true);
    wait_flag = true;
}

template <typename T>
FeatureMap<T>& FeatureMap<T>::operator=(const FeatureMap& rhs){
    if(this == &rhs)
        return *this;

    featureMap_size = rhs.featureMap_size;
    fmt = rhs.fmt;
    this->dim_h = rhs.dim_h;
    this->dim_w = rhs.dim_w;
    this->dim_z = rhs.dim_z;
    num_ch = rhs.num_ch;
    fmap_convert_threads_ = rhs.fmap_convert_threads_;
    formatted_featuremap_size = rhs.formatted_featuremap_size;
    if(fmap_data != NULL){
        delete[] fmap_data;
        fmap_data = NULL;
    }
    fmap_data = new T[featureMap_size];
    std::memcpy(fmap_data, rhs.fmap_data, featureMap_size);
    if(formatted_data != NULL && formatted_data != (uint8_t*) fmap_data){
        delete[] formatted_data;
        formatted_data = NULL;
    }
    if(fmt == MX_FMT_RGB888 || fmt == MX_FMT_FP32){
        formatted_data = (uint8_t*) fmap_data;
    } else {
        formatted_data = new uint8_t[formatted_featuremap_size];
        std::memcpy(formatted_data, rhs.formatted_data, formatted_featuremap_size);
    }
    return *this;
    out_ready.store(true);
    in_ready.store(true);
    wait_flag = true;
}


template <typename T>
void FeatureMap<T>::calc_convert_size_and_new()
{
    switch(fmt)
    {
        case MX_FMT_RGB888:
            // don't actually do anything
            formatted_featuremap_size = featureMap_size;
            formatted_data = (uint8_t*) fmap_data;
            break;
        case MX_FMT_FP32:
            // plain old *4
            formatted_featuremap_size = featureMap_size * 4;
            // the cast from float to uint8 accounts for the *4 size
            formatted_data = (uint8_t*) fmap_data;
            break;
        case MX_FMT_BF16:
            // extra padding item for odd-sized fmaps
            formatted_featuremap_size = featureMap_size * 2;
            if ( featureMap_size % 2 )
                formatted_featuremap_size += 2;
            // have to actually allocate this one
            formatted_data = new uint8_t[formatted_featuremap_size];
            break;
        case MX_FMT_GBF80: {
            // need to get fancy for this one...
            size_t num_xyz_pixels = (featureMap_size / num_ch);
            bool   any_remainder_chs = ((num_ch % 8) != 0);
            size_t num_gbf_words = (num_ch / 8) + (any_remainder_chs ? 1 : 0);
            formatted_featuremap_size = num_xyz_pixels * num_gbf_words * 10;
            // have to actually allocate this one
            formatted_data = new uint8_t[formatted_featuremap_size];
            break;
        }
        case MX_FMT_GBF80_ROW: {
            // need to get fancy for this one...
            bool   any_remainder_chs = ((num_ch % 8) != 0);
            size_t num_gbf_words = (num_ch / 8) + (any_remainder_chs ? 1 : 0);

            formatted_featuremap_size = this->dim_h * ((this->dim_w * this->dim_z * num_gbf_words * 10 + 3) & ~0x3);// padding row size to 4 bytes-alignment
            // have to actually allocate this one
            formatted_data = new uint8_t[formatted_featuremap_size];
            break;
        }
        default:
            throw std::invalid_argument("Invalid featureMap data format");
            break;
    }
}

template <typename T>
void FeatureMap<T>::convert_data() const
{
    if (fmt == MX_FMT_BF16)
    {
        uint32_t *fp_uint32 = (uint32_t*) fmap_data;
        
        #pragma omp for schedule(static)  // ignored if not parallel
        for(size_t i = 0; i < featureMap_size; i++){
            uint32_t v = fp_uint32[i] + 0x00008000;
            memcpy(&(formatted_data[i*2]), ((uint8_t*) &v)+2, 2);
        }
    }
    else if (fmt == MX_FMT_GBF80 || fmt == MX_FMT_GBF80_ROW)
    {
        size_t num_xyz_pixels = (featureMap_size / num_ch);
        bool   any_remainder_chs = ((num_ch % 8) != 0);
        size_t num_gbf_per_pixel = (num_ch / 8) + (any_remainder_chs ? 1 : 0);

        #pragma omp for schedule(static)  // ignored if not parallel
        for(size_t i = 0; i < num_xyz_pixels; i++){
            uint8_t *gbf_base = &(formatted_data[ i * (num_gbf_per_pixel * 10) ]);
            float   *flt_base = (float*) &(fmap_data[ i * num_ch ]);

            gbf_encode(flt_base, gbf_base, num_ch);
        }
    }
}


template <typename T>
void FeatureMap<T>::unconvert_data() const
{
    if (fmt == MX_FMT_BF16)
    {
        memset(fmap_data, 0, featureMap_size*sizeof(float)); // wipe fmap_data

        uint16_t *bf_dat = (uint16_t*) formatted_data;
        #pragma omp for schedule(static)  // ignored if not parallel
        for(size_t i = 0; i < featureMap_size; i++){
            memcpy(((uint8_t*) (fmap_data + i))+2, (uint8_t*) (bf_dat + i), 2);
        }
    }
    else if (fmt == MX_FMT_GBF80)
    {
        size_t num_xyz_pixels = (featureMap_size / num_ch);
        bool   any_remainder_chs = ((num_ch % 8) != 0);
        size_t num_gbf_per_pixel = (num_ch / 8) + (any_remainder_chs ? 1 : 0);

        #pragma omp for schedule(static)  // ignored if not parallel
        for(size_t i = 0; i < num_xyz_pixels; i++){
            uint8_t *gbf_base = &(formatted_data[ i * (num_gbf_per_pixel * 10) ]);
            float   *flt_base = (float*) &(fmap_data[ i * num_ch ]);

            gbf_decode(gbf_base, flt_base, num_ch);
        }
    }
    else if (fmt == MX_FMT_GBF80_ROW)
    {
        int num_gbf_per_pixel = (num_ch / 8) + (((num_ch%8)!=0) ? 1 : 0);
        int gbf80_pixel_size = num_gbf_per_pixel * 10;
        int gbf80_row_size = ((this->dim_w * this->dim_z * gbf80_pixel_size) + 3) & ~0x3;
        int flt32_row_size = this->dim_w  * this->dim_z * num_ch;
        size_t gbf80_row_offset = 0;
        size_t flt32_row_offset = 0;

        for (uint16_t height = 0; height < this->dim_h; height++) {
            size_t gbf80_pixel_offset = 0;
            size_t flt32_pixel_offset = 0;
            for (int width = 0; width < this->dim_w; width++) {
                for (int z = 0; z < this->dim_z; z++) {
                    uint8_t *gbf80_buffer = (uint8_t *)(formatted_data + gbf80_row_offset + gbf80_pixel_offset);
                    float *flt32_buffer = (float *)(fmap_data + flt32_row_offset + flt32_pixel_offset);
                    gbf_decode(gbf80_buffer, flt32_buffer, num_ch);

                    gbf80_pixel_offset += gbf80_pixel_size;
                    flt32_pixel_offset += num_ch;
                }
            }
            gbf80_row_offset += gbf80_row_size;
            flt32_row_offset += flt32_row_size;
        }
    }
}

template <typename T>
void FeatureMap<T>::transpose_hwc_chw(T* input, T* output) const {
    #pragma omp for schedule(static)  // ignored if not parallel
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

template <typename T>
void FeatureMap<T>::transpose_chw_hwc(T* input, T* output) const {
    #pragma omp for schedule(static)  // ignored if not parallel
    for (int c = 0; c < num_ch; ++c) {
        for (int h = 0; h < dim_h; ++h) {
            for (int w = 0; w < dim_w; ++w) {
                for (int d = 0; d < dim_z; ++d) {
                    output[h * dim_w * dim_z * num_ch + w * dim_z * num_ch + d * num_ch + c] =
                        input[c * dim_h * dim_w * dim_z + h * dim_w * dim_z + w * dim_z + d];

                }
            }
        }
    }
}

template<typename T>
T* FeatureMap<T>::get_data_ptr(){
    return fmap_data;
}

template <typename T>
MX_status FeatureMap<T>::get_data(T *out_data, bool channel_first) const
{
    if(fm_type!=FM_DFP){
        get_data_len(out_data);
        return MX_STATUS_OK;
    }

    #pragma omp parallel if(fmap_convert_threads_ > 1) num_threads(fmap_convert_threads_)
    {
        unconvert_data();

        if(channel_first){
            this->transpose_hwc_chw(fmap_data, out_data);
        }
        else{
            std::memcpy(out_data,fmap_data, featureMap_size*sizeof(T));
        }
    }
    return MX_STATUS_OK;
}

template <typename T>
MX_status FeatureMap<T>::set_data(T *in_data, bool channel_first) const
{
    if(fm_type!=FM_DFP){
        set_data_len(in_data);
        return MX_STATUS_OK;
    }

    #pragma omp parallel if(fmap_convert_threads_ > 1) num_threads(fmap_convert_threads_)
    {
        if(channel_first){
            this->transpose_chw_hwc(in_data,fmap_data);
        }
        else{
            std::memcpy(fmap_data, in_data, featureMap_size*sizeof(T));
        }
        convert_data();
    }
    in_ready.store(false);
    return MX_STATUS_OK;
}

template <typename T>
void FeatureMap<T>::set_data_len(T *in_data, size_t data_len) const
{
    if(data_len==0){
        std::memcpy(fmap_data, in_data, featureMap_size*sizeof(T));
    }
    else{
        std::memcpy(fmap_data, in_data, data_len*sizeof(T));
    }
}

template <typename T>
void FeatureMap<T>::get_data_len(T *out_data, size_t data_len) const
{
    if(data_len==0){
        std::memcpy(out_data,fmap_data, featureMap_size*sizeof(T));
    }
    else{
        std::memcpy(out_data,fmap_data, data_len*sizeof(T));
    }
}

template <typename T>
FeatureMap<T>::~FeatureMap()
{
    if (fmap_data != NULL)
    {
        delete[] fmap_data;
       fmap_data = NULL;
    }
    if (formatted_data != NULL)
    {
        // check so we don't don't double-free!
        if (fmt == MX_FMT_BF16 || fmt == MX_FMT_GBF80 || fmt == MX_FMT_GBF80_ROW)
        {
            delete[] formatted_data;
        }
        formatted_data = NULL;
    }
}

template <typename T>
uint8_t *FeatureMap<T>::get_formatted_data()
{
    return formatted_data;
}

template <typename T>
size_t FeatureMap<T>::get_formatted_size()
{
    return formatted_featuremap_size;
}

template <typename T>
std::vector<int64_t> FeatureMap<T>::shape(bool channel_first) const{
    MX::Types::ShapeVector shape_vec(dim_h, dim_w, dim_z, num_ch);
    if(channel_first){
        return shape_vec.chfirst_shape();
    }
    else{
        return shape_vec.chlast_shape();
    }

}

template <typename T>
void FeatureMap<T>::set_in_ready(bool flag){
    in_ready.store(flag);
}

template <typename T>
void FeatureMap<T>::set_out_ready(bool flag){
    out_ready.store(flag);
}

template <typename T>
bool FeatureMap<T>::get_out_ready(){
    return out_ready.load();
}

template <typename T>
bool FeatureMap<T>::get_in_ready(){
    return in_ready.load();
}

template <typename T>
int FeatureMap<T>::get_num_fmap_threads() const{
    return fmap_convert_threads_;
}

// template class FeatureMap<uint8_t>;
template class FeatureMap<float>;
