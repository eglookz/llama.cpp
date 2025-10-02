#pragma once

#include <memory>
#include <iostream>

#include "llama-model.h" 

class llama_layer_wrapper : public llama_layer {
public:
    explicit llama_layer_wrapper(int index = -1);
    ~llama_layer_wrapper();

private:
    int m_index;
};
