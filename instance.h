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
template <typename T> [[nodiscard]] Instance<T> *allocInstance(unsigned num_streams);

template <typename T> class BaseInstance {
  using Item = std::pair<Pipeline<T>, std::atomic_flag>;
  boost::sync::semaphore semaphore;

  Item *items() noexcept { return reinterpret_cast<Item *>(reinterpret_cast<unsigned *>(this + 1) + 1); }
  unsigned num_streams() const noexcept { return *reinterpret_cast<const unsigned *>(this + 1); }

  friend Instance<T> *allocInstance<>(unsigned num_streams);

protected:
  template <typename... Args1, typename... Args2>
  BaseInstance(std::tuple<Args1...> primaryPipelineArgs, std::tuple<Args2...> secondaryPipelineAdditionalArgs) : semaphore(num_streams()) {
    auto items = this->items();
    new (items) Item(std::piecewise_construct, primaryPipelineArgs, std::forward_as_tuple());
    items[0].second.clear();
    for (unsigned i = 1; i < num_streams(); ++i) {
      new (items + i) Item(std::piecewise_construct, std::tuple_cat(std::forward_as_tuple(firstReactor()), secondaryPipelineAdditionalArgs),
                           std::forward_as_tuple());
      items[i].second.clear();
    }
  }

public:
  ~BaseInstance() {
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i)
      items[i].~Item();
  }

  Pipeline<T> &firstReactor() { return items()[0].first; }

  Pipeline<T> &acquireReactor() {
    if (num_streams() == 1)
      return firstReactor();
    semaphore.wait();
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i) {
      if (!items[i].second.test_and_set())
        return items[i].first;
    }
    unreachable();
  }

  void releaseReactor(const Pipeline<T> &instance) {
    if (num_streams() == 1)
      return;
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i) {
      if (&instance == &items[i].first) {
        items[i].second.clear();
        break;
      }
    }
    semaphore.post();
  }
};

template <typename T> [[nodiscard]] Instance<T> *allocInstance(unsigned num_streams) {
  auto p =
      static_cast<Instance<T> *>(::operator new(sizeof(Instance<T>) + sizeof(unsigned) + sizeof(typename Instance<T>::Item) * num_streams));
  *reinterpret_cast<unsigned *>(p + 1) = num_streams;
  return p;
}
} // namespace
