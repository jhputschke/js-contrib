/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Helpers for the trampolines of Python modules (docs/Plans/PLAN_pyjetscape_complete.md):
 *
 *   - the containers C++ passes by reference reach Python as plain lists: the
 *     input partons as references to the C++ objects (changed in place), the
 *     output as a list the module appends to, copied back after the call;
 *   - Clone() of a Python module: a new instance of the same Python class made
 *     by the C++ copy constructor of the base (as a C++ module's implicit copy
 *     calls it: same id, settings and task-number registration order), then the
 *     instance attributes copied (shallow), as a C++ memberwise copy.
 ******************************************************************************/

#ifndef PYJS_MODULES_H
#define PYJS_MODULES_H

#include <pybind11/pybind11.h>

#include <memory>
#include <vector>

#include "JetScapeParticles.h"

namespace pyjs {

namespace py = pybind11;

// The C++ objects themselves (no copy): valid during the call only.
template <class T> py::list refs(std::vector<T> &v) {
  py::list out;
  for (auto &x : v)
    out.append(py::cast(&x, py::return_value_policy::reference));
  return out;
}

// Copies of the elements.
template <class T> py::list copies(const std::vector<T> &v) {
  py::list out;
  for (const auto &x : v)
    out.append(py::cast(T(x)));
  return out;
}

// After the call: an element the module replaced is copied in; a list whose
// length changed replaces the vector.
template <class T> void write_back(const py::list &l, std::vector<T> &v) {
  if (py::len(l) != v.size()) {
    std::vector<T> nv;
    nv.reserve(py::len(l));
    for (auto h : l)
      nv.push_back(h.cast<T>());
    v.swap(nv);
    return;
  }
  for (size_t i = 0; i < v.size(); ++i) {
    T &x = l[i].cast<T &>();
    if (&x != &v[i])
      v[i] = x;
  }
}

// The module's output list into the C++ vector (copies; a Photon is sliced to a
// Parton as a C++ module's push_back of a Photon does).
template <class T> void assign(const py::list &l, std::vector<T> &v) {
  v.clear();
  v.reserve(py::len(l));
  for (auto h : l)
    v.push_back(h.cast<T>());
}

// For shared_ptr containers (hadron and parton pointer lists): the same objects.
template <class T>
py::list ptrs(const std::vector<std::shared_ptr<T>> &v) {
  py::list out;
  for (const auto &p : v)
    out.append(py::cast(p));
  return out;
}

template <class T>
void assign_ptrs(const py::list &l, std::vector<std::shared_ptr<T>> &v) {
  v.clear();
  v.reserve(py::len(l));
  for (auto h : l)
    v.push_back(h.cast<std::shared_ptr<T>>());
}

// Clone() of a Python subclass of Base (Base bound with an
// init_alias<const Base &> copy constructor).
template <class Base> std::shared_ptr<Base> default_clone(const Base *self) {
  py::object me = py::cast(const_cast<Base *>(self));
  py::object cls = py::type::of(me);
  py::object copy = cls.attr("__new__")(cls);
  py::type::of<Base>().attr("__init__")(copy, me);
  if (py::hasattr(me, "__dict__"))
    copy.attr("__dict__").attr("update")(me.attr("__dict__"));
  return copy.cast<std::shared_ptr<Base>>();
}

} // namespace pyjs

#endif
