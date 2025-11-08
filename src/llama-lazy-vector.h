#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <cstddef>

template <typename T>
class llama_lazy_vector {
public:
    using loader_func = std::function<std::unique_ptr<T>(int index)>;

    llama_lazy_vector() = default;

    llama_lazy_vector(size_t size, loader_func loader)
        : size_(size), loader_(std::move(loader)) {
        items_.resize(size_);
    }

    void init(size_t size, loader_func loader) {
        size_ = size;
        loader_ = std::move(loader);
        items_.clear();
        items_.resize(size_);
    }

    void resize(size_t new_size) {
        std::cout << ">>>>>>> llama_layer new size = " << new_size << "\n";
        size_ = new_size;
        items_.resize(size_);
    }

    T& operator[](size_t index) {
        // Никаких автоматических выгрузок здесь: управление выгрузкой вне контейнера
        ensure_loaded(index);
        return *items_.at(index);
    }

    const T& operator[](size_t index) const {
        ensure_loaded(index);
        return *items_.at(index);
    }

    size_t size() const { return size_; }

    class Iterator {
    public:
        Iterator(llama_lazy_vector& vec, size_t pos) : vec_(vec), pos_(pos) {}
        T& operator*() { return vec_[pos_]; }
        Iterator& operator++() { ++pos_; return *this; }
        bool operator!=(const Iterator& other) const { return pos_ != other.pos_; }
    private:
        llama_lazy_vector& vec_;
        size_t pos_;
    };

    Iterator begin() { return Iterator(*this, 0); }
    Iterator end()   { return Iterator(*this, size_); }

    class ConstIterator {
    public:
        ConstIterator(const llama_lazy_vector& vec, size_t pos) : vec_(vec), pos_(pos) {}
        const T& operator*() const { return vec_[pos_]; }
        ConstIterator& operator++() { ++pos_; return *this; }
        bool operator!=(const ConstIterator& other) const { return pos_ != other.pos_; }
    private:
        const llama_lazy_vector& vec_;
        size_t pos_;
    };

    ConstIterator begin() const { return ConstIterator(*this, 0); }
    ConstIterator end()   const { return ConstIterator(*this, size_); }

    // В динамическом RAM режиме контейнер не занимается выгрузкой — no-op.
    void unload(size_t index) const { (void) index; }

    // Принудительная полная выгрузка (удаление структуры слоя)
    void unload_complete(size_t index) const {
        if (index < items_.size()) {
            std::cout << ">>> Complete unload of layer " << index << "\n";
            items_[index].reset();
        }
    }

private:
    void ensure_loaded(size_t index) const {
        if (!(index < size_)) {
            throw std::out_of_range("Index out of bounds");
        }
        if (!loader_) {
            throw std::runtime_error("Loader is not set");
        }

        if (index >= items_.size()) {
            items_.resize(size_);
        }

        if (!items_[index]) {
            items_[index] = loader_(static_cast<int>(index));
            if (!items_[index]) {
                throw std::runtime_error("Loader returned null pointer");
            }
        }
    }

    size_t size_ = 0;
    loader_func loader_ = nullptr;
    mutable std::vector<std::unique_ptr<T>> items_;
};
