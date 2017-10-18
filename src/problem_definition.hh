#ifndef _problem_definition_hh
#define _problem_definition_hh

#include <dune/pdelab/localoperator/nonlinearconvectiondiffusionfem.hh>


//! base class for parameter class
template<typename GV, typename RF>
class ConvectionDiffusionProblem :
  public Dune::PDELab::ConvectionDiffusionParameterInterface<
  Dune::PDELab::ConvectionDiffusionParameterTraits<GV,RF>,
  ConvectionDiffusionProblem<GV,RF>
  >
{
public:
  typedef Dune::PDELab::ConvectionDiffusionParameterTraits<GV,RF> Traits;

  //! source/reaction term
  typename Traits::RangeFieldType
  f (const typename Traits::ElementType& e, const typename Traits::DomainType& x ) const
  {
    typename Traits::RangeType global = e.geometry().global(x);
    typename Traits::RangeFieldType X = sin(3.0*M_PI*global[0]);
    typename Traits::RangeFieldType Y = sin(2.0*M_PI*global[1]);
    return 1.;
  }

  //! tensor permeability
  typename Traits::PermTensorType
  A (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType kabs;
    for (std::size_t i=0; i<Traits::dimDomain; i++)
      for (std::size_t j=0; j<Traits::dimDomain; j++)
        kabs[i][j] = (i==j) ? 1 : 0;
    return kabs;
  }

   //! boundary condition type function
  template<typename I>
  bool isDirichlet(
                   const I & intersection,               /*@\label{bcp:name}@*/
                   const Dune::FieldVector<typename I::ctype, I::dimension-1> & coord
                   ) const
  {

    //Dune::FieldVector<typename I::ctype, I::dimension>
    //  xg = intersection.geometry().global( coord );

    return true;  // Dirichlet b.c. on all boundaries
  }

  //! Dirichlet boundary condition value
  typename Traits::RangeFieldType
  g (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    return 0.0;
  }

  //! Neumann boundary condition
  typename Traits::RangeFieldType
  j (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    return 0.0;
  }

};

#endif
