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
        : size_(size), loader_(std::move(loader)) {
        items_.resize(size_);
    }

    // Метод инициализации (если использован дефолтный конструктор)
    void init(size_t size, loader_func loader) {
        size_ = size;
        loader_ = std::move(loader);
        items_.clear();
        items_.resize(size_);
    }

    // Изменение размера
    void resize(size_t new_size) {
    std::cout << ">>>>>>> llama_layer new size = " << new_size << "\n";
        // Прежняя версия уменьшала размер на 1 — это приводит к неверным
        // границам и потенциальным выходам за пределы (segfault).
        // Теперь сохраняем ожидаемый размер напрямую.
        size_ = new_size;
        // Уменьшаем или расширяем вектор указателей элементов.
        items_.resize(size_);
    }

    T& operator[](size_t index) {
        //// std::cout << "!!!!!!!!! llama_layer[" << index << "] was accessed\n";
        //// std::cout << "!!!!!!!!! prev current_index = " << current_index_ << "\n";
        if (index > 2) {
            unload(index - 1);
        }
        ensure_loaded(index);
        return *items_.at(index);
    }

    // const version — только если слой уже загружен!
    const T& operator[](size_t index) const {
        //// std::cout << "llama_layer[" << index << "] was accessed\n";
        //// std::cout << "prev current index = " << current_index_ << "\n";
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

    // Освободить конкретный индекс (явная выгрузка элемента)
    // Безопасно вызывать снаружи вместо попытки освобождать из деструктора элемента
    void unload(size_t index) const {
        // Проверка нижней и верхней границ: допустимо [0, size_)
        // (для size_t индекс не бывает < 0, но сохраняем явную проверку верхней границы)
        if (!(index < size_)) {
            throw std::out_of_range("unload: index out of bounds");
        }
        if (index < items_.size()) {
            // items_[index].reset();

            // Освобождаем основные веса attention
            // items_[index]->wq = nullptr;
            // items_[index]->wk = nullptr;
            // items_[index]->wv = nullptr;
            // items_[index]->wo = nullptr;

            // Biases (если есть)
            // items_[index]->bq = nullptr;
            // items_[index]->bk = nullptr;
            // items_[index]->bv = nullptr;
            // items_[index]->bo = nullptr;

            // // Нормализации
            // items_[index]->attn_norm = nullptr;
            // items_[index]->ffn_norm = nullptr;

            // // FFN веса
            // items_[index]->ffn_gate = nullptr;
            // items_[index]->ffn_down = nullptr;
            // items_[index]->ffn_up = nullptr;
            // // или для MoE
            // items_[index]->ffn_gate_inp = nullptr;
            // items_[index]->ffn_gate_exps = nullptr;
            // items_[index]->ffn_down_exps = nullptr;
            // items_[index]->ffn_up_exps = nullptr;
            
            // // ROPE тензоры удаляем ТОЛЬКО для первого слоя
            // if (index == 0) {
            //     items_[index]->rope_freqs = nullptr;
            //     items_[index]->rope_long = nullptr;
            //     items_[index]->rope_short = nullptr;
            // }
            // // Для остальных слоёв НЕ трогаем - они указывают на данные из слоя 0!
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
        // if element not yet created, call loader and store the unique_ptr
        if (index >= items_.size() || !items_.at(index)) {
            if (index >= items_.size()) {
                // items_ may be smaller than logical size_ if nothing has been
                // loaded yet; ensure it has at least size_ elements so we can
                // store at the requested index (index < size_ is guaranteed).
                items_.resize(size_);
            }
            items_[index] = loader_(static_cast<int>(index));
            if (!items_[index]) {
                throw std::runtime_error("Loader returned null pointer");
            }
        }
    }

    size_t size_ = 0;
    loader_func loader_ = nullptr;

    // Храним по одному указателю на каждый элемент — это предотвращает
    // уничтожение ранее созданных объектов при загрузке нового элемента.
    // Это важно: построение графа может делать указания на созданные
    // объекты, поэтому они не должны разрушаться, пока граф их использует.
    mutable std::vector<std::unique_ptr<T>> items_;
};
