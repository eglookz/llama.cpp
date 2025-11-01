#include "llama-layer-wrapper.h"

llama_layer_wrapper::llama_layer_wrapper(int index)
    : m_index(index)
{
    //// std::cout << "llama_layer[" << m_index << "] was created\n";
}

llama_layer_wrapper::~llama_layer_wrapper() {
    // std::cout << "llama_layer[" << m_index << "] was destroyed\n";
}
