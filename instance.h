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
#include <type_traits>
#include <utility>

#include <boost/lockfree/policies.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/sync/semaphore.hpp>

#include "common.h"

namespace {
template <typename T> class Pipeline;
template <typename T> class Instance;
template <typename T> [[nodiscard]] Instance<T> *allocInstance(std::size_t num_streams);

template <typename T> class BaseInstance {
  static constexpr unsigned max_num_streams = 32;

  boost::sync::semaphore semaphore;
  boost::lockfree::queue<unsigned, boost::lockfree::capacity<max_num_streams>> available;

  Pipeline<T> *items() noexcept { return reinterpret_cast<Pipeline<T> *>(reinterpret_cast<std::size_t *>(this + 1) + 1); }
  unsigned num_streams() const noexcept { return static_cast<unsigned>(*reinterpret_cast<const std::size_t *>(this + 1)); }

  friend Instance<T> *allocInstance<>(std::size_t num_streams);

  template <typename... Args, size_t... Indexes>
  void initPipeline(Pipeline<T> *p, std::tuple<Args...> args, std::index_sequence<Indexes...>) {
    new (p) Pipeline<T>(std::get<Indexes>(std::move(args))...);
  }

protected:
  template <typename... Args1, typename... Args2>
  BaseInstance(std::tuple<Args1...> primaryPipelineArgs, std::tuple<Args2...> secondaryPipelineAdditionalArgs)
      : semaphore(num_streams()) {
    if (num_streams() > max_num_streams) {
      throw std::runtime_error("too many streams");
    }

    auto items = this->items();
    initPipeline(items, primaryPipelineArgs, std::index_sequence_for<Args1...>{});
    available.push(0);
    for (unsigned i = 1; i < num_streams(); ++i) {
      initPipeline(items + i, std::tuple_cat(std::forward_as_tuple(firstReactor()), secondaryPipelineAdditionalArgs),
                   std::make_index_sequence<sizeof...(Args2) + 1>{});
      available.push(i);
    }
  }

public:
  ~BaseInstance() {
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i)
      items[i].~Pipeline<T>();
  }

  Pipeline<T> &firstReactor() { return items()[0]; }

  Pipeline<T> &acquireReactor() {
    if (num_streams() == 1)
      return firstReactor();
    semaphore.wait();
    auto items = this->items();
    unsigned i;
    available.pop(i);
    return items[i];
  }

  void releaseReactor(const Pipeline<T> &reactor) {
    if (num_streams() == 1)
      return;
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i) {
      if (&reactor == items + i) {
        available.push(i);
        break;
      }
    }
    semaphore.post();
  }
};

template <typename T> [[nodiscard]] Instance<T> *allocInstance(std::size_t num_streams) {
  constexpr std::size_t fr = sizeof(Instance<T>) + sizeof(num_streams);
  static_assert(__STDCPP_DEFAULT_NEW_ALIGNMENT__ % alignof(Pipeline<T>) == 0 && fr % alignof(Pipeline<T>) == 0,
                "unable to allocate instance with proper alignment");
  auto p = static_cast<Instance<T> *>(::operator new(fr + sizeof(Pipeline<T>) * num_streams));
  *reinterpret_cast<std::size_t *>(p + 1) = num_streams;
  return p;
}
} // namespace
