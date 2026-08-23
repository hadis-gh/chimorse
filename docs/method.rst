Scientific method
=================

This page summarizes the model construction implemented in ChiMorse. It is a
software-oriented guide, not a replacement for the derivation and validation in
the accompanying manuscript.

Reference interaction landscape
-------------------------------

The framework starts from sampled pair-interaction profiles
:math:`E_{\mathrm{ref}}(r,\varphi_1,\varphi_2)` for periodic orientational
coordinates :math:`\varphi_1` and :math:`\varphi_2`. Each sampled angular
configuration should include a radial profile covering the physically relevant
repulsive and attractive regions so that the local equilibrium separation,
well depth, and radial shape can be determined.

Collective angular coordinates
------------------------------

For the chiral-helix formulation used in the reference workflow, the molecular
rotation angles are transformed to

.. math::

   \chi = \varphi_1 - h\varphi_2,
   \qquad
   \psi = \varphi_1 + h\varphi_2,

where :math:`h` represents the class-dependent handedness/screw-direction sign.
In this representation, :math:`\chi` describes relative angular registry and
:math:`\psi` the joint angular phase. The screw and interchange symmetries of
the demonstration system can then be translated into restrictions on the
allowed Fourier terms.

Radial Morse model
------------------

At each angular configuration the radial interaction is represented by

.. math::

   V(r;\chi,\psi) = D(\chi,\psi)
   \left[
   e^{-2\alpha(\chi,\psi)(r-r_e(\chi,\psi))}
   -2e^{-\alpha(\chi,\psi)(r-r_e(\chi,\psi))}
   \right].

The well depth :math:`D` and equilibrium separation :math:`r_e` are extracted
from the local radial minimum. ChiMorse can either use a compact form with a
fixed radial-width parameter :math:`\alpha`, or fit and represent
:math:`\alpha(\chi,\psi)` as an additional angular field.

Symmetry-adapted Fourier representation
---------------------------------------

The angular dependence of the Morse parameters is expanded in a two-dimensional
Fourier basis. Known invariances are imposed *before fitting* by excluding
symmetry-incompatible terms. In the alphaPA demonstration, molecular screw
symmetry restricts the allowed harmonics along :math:`\psi`, while interchange
symmetry imposes even parity in :math:`\chi` for the parallel interaction
classes. The admissible basis therefore depends on the interaction class.

This exact symmetry reduction is distinct from approximate compression: it
removes coefficients that are forbidden by the physical invariances rather than
coefficients that merely happen to be small in one fitted dataset.

Parameterization and reduction
------------------------------

The implemented workflow is:

#. Extract local :math:`D(\chi,\psi)` and :math:`r_e(\chi,\psi)` from the
   reference radial profiles; optionally fit :math:`\alpha(\chi,\psi)`.
#. Construct the symmetry-adapted Fourier design matrix.
#. Determine Fourier coefficients by linear least squares for fixed harmonic
   orders.
#. Increase the retained harmonic resolution to assess convergence.
#. Select a sufficiently converged harmonic basis.
#. Optionally prune weak symmetry-allowed coefficients and refit the retained
   terms.
#. Evaluate the analytical model on the reference grid and quantify energy and
   near-equilibrium errors.
#. Export fitted coefficients when the potential is to be evaluated by an
   external simulation code.

Model reduction therefore occurs at three conceptually different levels:
physical symmetry restrictions, harmonic truncation, and coefficient pruning.
Keeping those stages separate makes the final model easier to interpret and the
accuracy--compactness trade-off easier to quantify.

Reference interaction classes
-----------------------------

The alphaPA demonstration distinguishes four classes:

``EP``
   Equal handedness, parallel axial alignment.
``EA``
   Equal handedness, antiparallel axial alignment.
``OP``
   Opposite handedness, parallel axial alignment.
``OA``
   Opposite handedness, antiparallel axial alignment.

Each class is modeled independently unless an exact symmetry operation relates
configurations within that class. The alphaPA-specific symmetry choices should
not be copied to another physical system without a corresponding symmetry
analysis.
