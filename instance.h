/*
 * EEDI2CUDA: EEDI2 filter using CUDA
 *
 * Copyright (C) 2005-2006 Kevin Stone
 * Copyright (C) 2014-2019 HolyWu
 * Copyright (C) 2021 Misaki Kasumi
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#pragma once

#include <atomic>
#include <tuple>
#include <utility>

#include <boost/sync/semaphore.hpp>

#include "common.h"

namespace {
template <typename T> class Pipeline;
template <typename T> class Instance;
template <typename T> [[nodiscard]] Instance<T> *allocInstance(std::size_t num_streams);

template <typename T> class BaseInstance {
  using Item = std::pair<Pipeline<T>, std::atomic_flag>;
  boost::sync::semaphore semaphore;

  Item *items() noexcept { return reinterpret_cast<Item *>(reinterpret_cast<std::size_t *>(this + 1) + 1); }
  std::size_t num_streams() const noexcept { return *reinterpret_cast<const std::size_t *>(this + 1); }

  friend Instance<T> *allocInstance<>(std::size_t num_streams);

protected:
  template <typename... Args1, typename... Args2>
  BaseInstance(std::tuple<Args1...> primaryPipelineArgs, std::tuple<Args2...> secondaryPipelineAdditionalArgs)
      : semaphore(boost::numeric_cast<unsigned>(num_streams())) {
    auto items = this->items();
    new (items) Item(std::piecewise_construct, primaryPipelineArgs, std::forward_as_tuple());
    items[0].second.clear();
    for (std::size_t i = 1; i < num_streams(); ++i) {
      new (items + i) Item(std::piecewise_construct, std::tuple_cat(std::forward_as_tuple(firstReactor()), secondaryPipelineAdditionalArgs),
                           std::forward_as_tuple());
      items[i].second.clear();
    }
  }

public:
  ~BaseInstance() {
    auto items = this->items();
    for (std::size_t i = 0; i < num_streams(); ++i)
      items[i].~Item();
  }

  Pipeline<T> &firstReactor() { return items()[0].first; }

  Pipeline<T> &acquireReactor() {
    if (num_streams() == 1)
      return firstReactor();
    semaphore.wait();
    auto items = this->items();
    auto base = rand() % num_streams();
    for (std::size_t j = 0; j < num_streams(); ++j) {
      auto i = (base + j) % num_streams();
      if (!items[i].second.test_and_set(std::memory_order_acquire))
        return items[i].first;
    }
    unreachable();
  }

  void releaseReactor(const Pipeline<T> &reactor) {
    if (num_streams() == 1)
      return;
    auto items = this->items();
    for (std::size_t i = 0; i < num_streams(); ++i) {
      if (&reactor == &items[i].first) {
        items[i].second.clear(std::memory_order_release);
        break;
      }
    }
    semaphore.post();
  }
};

template <typename T> [[nodiscard]] Instance<T> *allocInstance(std::size_t num_streams) {
  typedef typename Instance<T>::Item Item;
  constexpr std::size_t fr = sizeof(Instance<T>) + sizeof(num_streams);
  static_assert((__STDCPP_DEFAULT_NEW_ALIGNMENT__ + fr) % alignof(Item) == 0, "unable to allocate instance with proper alignment");
  auto p = static_cast<Instance<T> *>(::operator new(fr + sizeof(Item) * num_streams));
  *reinterpret_cast<std::size_t *>(p + 1) = num_streams;
  return p;
}
} // namespace
