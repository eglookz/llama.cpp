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
        // Выгружаем предыдущий слой (кроме слоя 0)
        if (index > 2) {
            unload(index - 1);
        }
        
        // ensure_loaded перезагрузит слой, если он был выгружен
        ensure_loaded(index);
        return *items_.at(index);
    }

    const T& operator[](size_t index) const {
        if (index > 2) {
            unload(index - 1);
        }
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

    // Освобождаем только тяжёлые веса, но оставляем структуру слоя
    void unload(size_t index) const {
        if (!(index < size_)) {
            return; // Безопасно игнорируем
        }
        
        if (index >= items_.size() || !items_[index]) {
            return; // Слой уже выгружен
        }

        // std::cout << ">>> Unloading layer " << index << "\n";

        // НЕ ДЕЛАЕМ items_[index].reset()!
        // Вместо этого обнуляем только тяжёлые веса
        
        // Attention веса (самые тяжёлые)
        items_[index]->wq = nullptr;
        items_[index]->wk = nullptr;
        items_[index]->wv = nullptr;
        items_[index]->wo = nullptr;

        // Biases
        items_[index]->bq = nullptr;
        items_[index]->bk = nullptr;
        items_[index]->bv = nullptr;
        items_[index]->bo = nullptr;

        // Нормализации
        items_[index]->attn_norm = nullptr;
        items_[index]->ffn_norm = nullptr;

        // FFN веса
        items_[index]->ffn_gate = nullptr;
        items_[index]->ffn_down = nullptr;
        items_[index]->ffn_up = nullptr;
        
        // MoE веса (если есть)
        items_[index]->ffn_gate_inp = nullptr;
        items_[index]->ffn_gate_exps = nullptr;
        items_[index]->ffn_down_exps = nullptr;
        items_[index]->ffn_up_exps = nullptr;

        // ROPE тензоры НЕ трогаем (они shared для всех слоёв)
        // Только для слоя 0 они реально существуют
    }

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

        // КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: проверяем не только !items_[index],
        // но и загружен ли слой (проверяем основные веса)
        bool need_reload = !items_[index];
        
        if (items_[index]) {
            // Слой существует, но проверяем, не выгружены ли веса
            // Проверяем один из критичных тензоров
            if (!items_[index]->wq) {
                need_reload = true;
                // std::cout << ">>> Layer " << index << " weights unloaded, reloading...\n";
            }
        }

        if (need_reload) {
            // std::cout << ">>> Loading layer " << index << "\n";
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
