"""
python/jetscape/particles.py

A property view of one particle (docs/Plans/PLAN_pyjetscape_pythonic.md, step 7):

    v = view(p)
    v.pt, v.eta, v.e, v.momentum      # instead of p.pt(), p.eta(), p.e(), ...
    view(p).unwrap() is p

The bound classes keep their methods (p.pt()); properties there would replace them and
break every script that calls them.  Each property is one call into C++, about 1.4 us
(PLAN_pyjetscape_complete.md, V4): for many particles use the arrays
(final_parton_array(), hadron_array(), to_awkward()), not a loop over views.
"""

from __future__ import annotations

__all__ = ["ParticleView", "view"]


def _method(name: str, doc: str):
    def get(self):
        return getattr(self._p, name)()
    get.__name__ = name
    return property(get, doc=doc)


class ParticleView:
    """Read-only properties over a Parton, Hadron or Photon (any JetScapeParticleBase)."""

    __slots__ = ("_p",)

    def __init__(self, particle):
        self._p = particle

    def unwrap(self):
        """The C++ particle itself."""
        return self._p

    def __repr__(self):
        return f"view({self._p!r})"

    pid = _method("pid", "PDG id")
    pstat = _method("pstat", "status")
    label = _method("plabel", "label")
    e = _method("e", "energy [GeV]")
    px = _method("px", "px [GeV]")
    py = _method("py", "py [GeV]")
    pz = _method("pz", "pz [GeV]")
    pt = _method("pt", "transverse momentum [GeV]")
    eta = _method("eta", "pseudorapidity")
    phi = _method("phi", "azimuth in [0, 2 pi)")
    rapidity = _method("rapidity", "rapidity")
    mass = _method("restmass", "rest mass [GeV]")
    t = _method("t", "position time [fm/c]")
    x = _method("x", "x [fm]")
    y = _method("y", "y [fm]")
    z = _method("z", "z [fm]")

    @property
    def momentum(self):
        """FourVector, iterates as (E, px, py, pz)."""
        return self._p.p_in()

    @property
    def position(self):
        """FourVector, iterates as (t, x, y, z)."""
        return self._p.x_in()


def view(particle) -> ParticleView:
    """A ParticleView of `particle`."""
    return ParticleView(particle)
