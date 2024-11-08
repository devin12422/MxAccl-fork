#include <memx/accl/utils/mxTypes.h>

using namespace MX::Types;
             
// Constructor
MX::Types::ShapeVector::ShapeVector() : shape(4, 1) {} // Initialize shape with 4 elements, all initialized to 0
// Constructor
MX::Types::ShapeVector::ShapeVector(int64_t h, int64_t w, int64_t z, int64_t c) {
    this->shape.reserve(4);
    this->h = h;
    this->shape.push_back(h); // = h;
    this->w = w;
    this->shape.push_back(w); // = w;
    this->z = z;
    this->shape.push_back(z); // = z;
    this->c = c;
    this->shape.push_back(c); // = c;
    // std::cout<<"this shape size = "<<this->shape.size();
}
// Constructor
MX::Types::ShapeVector::ShapeVector(int size) : shape(size, 1), size_(size) {} // Initialize shape with 4 elements, all initialized to 0


// Overload [] operator 
// const int64_t& MX::Types::ShapeVector::operator[](int64_t index) const {
//     if (index < 4) {
//         return shape[index];
//     } else {
//         std::cerr << "Error: Index out of range." << std::endl;
//         throw std::runtime_error("Error: Index out of range." );
//     }
// }

// Overload [] operator 
int64_t& MX::Types::ShapeVector::operator[](int64_t index) {
    if (index < size_) {
        return shape[index];
    } else {
        std::cerr << "Error: Index out of range." << std::endl;
        throw std::runtime_error("Error: Index out of range." );
    }
}

std::vector<int64_t> MX::Types::ShapeVector::chfirst_shape(){
    std::vector chfirst_shape{this->z, this->c, this->h, this->w};
    return chfirst_shape;
}

std::vector<int64_t> MX::Types::ShapeVector::chlast_shape(){
    // std::vector chfirst_shape{this->h, this->w, this->z, this->c};
    return shape;
}             

// Function to return a pointer to shape vector
int64_t* MX::Types::ShapeVector::data() {
    return shape.data();
}

// Function to return the size of the internal vector
int64_t MX::Types::ShapeVector::size() const {
    return shape.size();
}

void MX::Types::ShapeVector::set_ch_first(){
    shape = this->chfirst_shape();
}
