#include <fastjet/ClusterSequence.hh>
#include <fastjet/contrib/SoftDrop.hh>
#include <fastjet/contrib/ConstituentSubtractor.hh>
#include <cstdio>
int main() {
  std::vector<fastjet::PseudoJet> p{{10, 0, 1, 10.05}, {8, 1, 0, 8.07}, {0.5, 0.2, 0.1, 0.55}};
  fastjet::ClusterSequence cs(p, fastjet::JetDefinition(fastjet::antikt_algorithm, 0.4));
  fastjet::contrib::SoftDrop sd(0.0, 0.1);
  auto j = sd(cs.inclusive_jets()[0]);
  fastjet::contrib::ConstituentSubtractor sub;
  printf("softdrop pt %.2f, %s\n", j.pt(), sub.description().substr(0, 40).c_str());
}
