# example/analysis_root/zjet: the C++ map steps of the Z+jet analyses (README.md).
#   make                     # with root-config and fastjet-config on PATH (conda env js_fno)
#   make ROOT_CONFIG=... FASTJET_CONFIG=...
#   make FASTJET_PY=$(python -c 'import fastjet, os; print(os.path.dirname(fastjet.__file__))')
#                            # the FastJet of a pip fastjet wheel: the Python map steps' own
#                            # version, with fjcontrib (libfastjetcontribfragile); the wheel's
#                            # fastjet-config has its build paths, so the flags are set here
ROOT_CONFIG    ?= root-config
FASTJET_CONFIG ?= fastjet-config
CXX            ?= g++
CXXFLAGS       += -O2 -Wall -Wextra $(shell $(ROOT_CONFIG) --cflags)
ROOTLIBS       := $(shell $(ROOT_CONFIG) --libs) -lROOTNTuple -lROOTDataFrame
ifdef FASTJET_PY
FJLIBDIR       := $(firstword $(wildcard $(FASTJET_PY)/lib64 $(FASTJET_PY)/lib))
FJFLAGS        := -I$(FASTJET_PY)/include
FJLIBS         := -L$(FJLIBDIR) -Wl,-rpath,$(FJLIBDIR) -lfastjetcontribfragile -lfastjettools \
                  -lfastjet
else
FJFLAGS        := $(shell $(FASTJET_CONFIG) --cxxflags)
FJLIBS         := $(shell $(FASTJET_CONFIG) --libs)
endif
RPATH          := -Wl,-rpath,$(shell $(ROOT_CONFIG) --libdir)

all: zhadron_pools zjets zqa

zhadron_pools: zhadron_pools.cxx ZJetCommon.h ../HadronFileReader.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(ROOTLIBS) $(RPATH)

zjets: zjets.cxx ZJetCommon.h ../HadronFileReader.h
	$(CXX) $(CXXFLAGS) $(FJFLAGS) -o $@ $< $(ROOTLIBS) $(FJLIBS) $(RPATH)

zqa: zqa.cxx ZJetCommon.h ../HadronFileReader.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(ROOTLIBS) $(RPATH)

clean:
	rm -f zhadron_pools zjets zqa

.PHONY: all clean
