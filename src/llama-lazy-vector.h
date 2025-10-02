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

    // Дефолтный конструктор (отложенная инициализация)
    llama_lazy_vector() = default;

    // Основной конструктор
    llama_lazy_vector(size_t size, loader_func loader)
        : size_(size), loader_(std::move(loader)), current_index_(-1) {}

    // Метод инициализации (если использован дефолтный конструктор)
    void init(size_t size, loader_func loader) {
        size_ = size;
        loader_ = std::move(loader);
        current_index_ = -1;
        current_.reset();
    }

    // Изменение размера
    void resize(size_t new_size) {
        std::cout << ">>>>>>> llama_layer new size = " << new_size << "\n";
        // Прежняя версия уменьшала размер на 1 — это приводит к неверным
        // границам и потенциальным выходам за пределы (segfault).
        // Теперь сохраняем ожидаемый размер напрямую.
        size_ = new_size;
        // Если текущий загруженный индекс вне новых границ — сбрасываем кеш.
        if (current_index_ >= static_cast<std::ptrdiff_t>(new_size)) {
            current_index_ = -1;
            current_.reset();
        }
    }

    T& operator[](size_t index) {
        std::cout << "!!!!!!!!! llama_layer[" << index << "] was accessed\n";
        std::cout << "!!!!!!!!! prev current_index = " << current_index_ << "\n";
        ensure_loaded(index);
        return *current_;
    }

    // const version — только если слой уже загружен!
    const T& operator[](size_t index) const {
        std::cout << "llama_layer[" << index << "] was accessed\n";
        std::cout << "prev current index = " << current_index_ << "\n";
        ensure_loaded(index);
        return *current_;
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

private:
    void ensure_loaded(size_t index) const {
        if (!(index < size_)) {
            throw std::out_of_range("Index out of bounds");
        }
        if (!loader_) {
            throw std::runtime_error("Loader is not set");
        }
        if (current_index_ != static_cast<std::ptrdiff_t>(index) || !current_) {
            current_ = loader_(static_cast<int>(index));
            if (!current_) {
                throw std::runtime_error("Loader returned null pointer");
            }
            current_index_ = static_cast<std::ptrdiff_t>(index);
            std::cout << "NEW current index = " << current_index_ << "\n";
        }
    }

    size_t size_ = 0;
    loader_func loader_ = nullptr;

    // Используем знаковый тип для хранения «-1» как маркера «нет загруженного">
    mutable std::ptrdiff_t current_index_ = -1;
    mutable std::unique_ptr<T> current_ = nullptr;
};
