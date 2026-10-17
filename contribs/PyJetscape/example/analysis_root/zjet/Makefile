# example/analysis_root/zjet: the C++ map steps of the Z+jet analyses (README.md).
#   make                     # with root-config and fastjet-config on PATH (conda env js_fno)
#   make ROOT_CONFIG=... FASTJET_CONFIG=...
ROOT_CONFIG    ?= root-config
FASTJET_CONFIG ?= fastjet-config
CXX            ?= g++
CXXFLAGS       += -O2 -Wall -Wextra $(shell $(ROOT_CONFIG) --cflags)
ROOTLIBS       := $(shell $(ROOT_CONFIG) --libs) -lROOTNTuple -lROOTDataFrame
FJFLAGS        := $(shell $(FASTJET_CONFIG) --cxxflags)
FJLIBS         := $(shell $(FASTJET_CONFIG) --libs)
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
