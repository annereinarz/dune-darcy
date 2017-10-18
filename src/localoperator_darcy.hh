// -*- tab-width: 4; c-basic-offset: 2; indent-tabs-mode: nil -*-

#ifndef DUNE_PDELAB_DARCY_HH
#define DUNE_PDELAB_DARCY_HH

#include<dune/pdelab/common/quadraturerules.hh>
#include<dune/pdelab/common/referenceelements.hh>
#include<dune/pdelab/localoperator/pattern.hh>
#include<dune/pdelab/localoperator/flags.hh>
#include<dune/pdelab/localoperator/idefault.hh>
#include<dune/pdelab/localoperator/defaultimp.hh>
#include<dune/pdelab/finiteelement/localbasiscache.hh>



namespace Dune {
    namespace PDELab {
        template<typename GV, typename PARAM, int dofel>
            class darcy : 
                public NumericalJacobianApplyVolume<darcy<GV,PARAM,dofel>>,
                public NumericalJacobianApplyBoundary<darcy<GV,PARAM,dofel>>,
                public FullVolumePattern,
                public LocalOperatorDefaultFlags,
                public InstationaryLocalOperatorDefaultMethods<double>,
                public Dune::PDELab::NumericalJacobianVolume<darcy<GV,PARAM,dofel> >,
                public Dune::PDELab::NumericalJacobianBoundary<darcy<GV,PARAM,dofel> >

        {
            public:
                // pattern assembly flags
                enum { doPatternVolume = true };

                // residual assembly flags
                enum { doAlphaVolume = true };
                enum { doAlphaBoundary = true };

                //constructor
                darcy (const GV gv_, PARAM param_, int intorder_=2) : gv(gv_), param(param_), intorder(intorder_){
                }


                // volume integral depending on test and ansatz functions
                template<typename EG, typename LFSU, typename X, typename LFSV, typename R>
                    void alpha_volume (const EG& eg, const LFSU& lfsu, const X& x, const LFSV& lfsv, R& r) const
                    {
                        // Define types
                        using RF = typename LFSU::Traits::FiniteElementType::
                            Traits::LocalBasisType::Traits::RangeFieldType;
                        using size_type = typename LFSU::Traits::SizeType;
                        typedef typename LFSV::Traits::FiniteElementType::Traits::LocalBasisType::Traits::
                            JacobianType JacobianType;
                        const unsigned int npe = lfsv.size();

                        // dimensions
                        const int dim = EG::Entity::dimension;

                        // Reference to cell
                        const auto& cell = eg.entity();

                        // Get geometry
                        auto geo = eg.geometry();

                        // evaluate diffusion tensor at cell center, assume it is constant over elements
                        auto ref_el = referenceElement(geo);
                        auto localcenter = ref_el.position(0,0);
                        auto tensor = param.A(cell,localcenter);

                        // Initialize vectors outside for loop
                        std::vector<Dune::FieldVector<RF,dim> > gradphi(lfsu.size());
                        Dune::FieldVector<RF,dim> gradu(0.0);
                        Dune::FieldVector<RF,dim> Agradu(0.0);

                        // Transformation matrix
                        typename EG::Geometry::JacobianInverseTransposed jac;

                        // loop over quadrature points
                        for (const auto& ip : quadratureRule(geo,intorder))
                        {
                            //Evaluate Shape functions
                            std::vector<Dune::FieldVector<RF,1>> phi(npe);
                            lfsv.finiteElement().localBasis().evaluateFunction(ip.position(),phi);

                            // Evaluate Jacobian
                            std::vector<JacobianType> js(npe);
                            lfsv.finiteElement().localBasis().evaluateJacobian(ip.position(),js);

                            // evaluate u
                            RF u=0.0;
                            for (size_type i=0; i<lfsu.size(); i++)
                                u += x(lfsu,i)*phi[i];

                            // transform gradients of shape functions to real element
                            jac = geo.jacobianInverseTransposed(ip.position());
                            for (size_type i=0; i<lfsu.size(); i++)
                                jac.mv(js[i][0],gradphi[i]);

                            // compute gradient of u
                            gradu = 0.0;
                            for (size_type i=0; i<lfsu.size(); i++)
                                gradu.axpy(x(lfsu,i),gradphi[i]);

                            // compute A * gradient of u
                            tensor.mv(gradu,Agradu);

                            // evaluate velocity field, sink term and source term
                            auto f = param.f(cell,ip.position());

                            // integrate (A grad u)*grad phi_i - u b*grad phi_i + c*u*phi_i
                            RF factor = ip.weight() * geo.integrationElement(ip.position());
                            //Accumulate the values at each quadrature point for each degree of freedom
                            //TODO instead of 0.0 add the term needed to evaluate gradphi * A * gradphi -f * phi
                            for (size_type i=0; i<lfsu.size(); i++)
                                r.accumulate(lfsu,i, ( Agradu*gradphi[i] - f*phi[i] )*factor);
                        }
                    } // end alpha_volume

                // boundary integral
                template<typename IG, typename LFSU, typename X, typename LFSV, typename R>
                    void alpha_boundary (const IG& ig,
                            const LFSU& lfsu_s, const X& x_s, const LFSV& lfsv_s, R& r_s) const
                    {
                        // Define types
                        using RF = typename LFSV::Traits::FiniteElementType::
                            Traits::LocalBasisType::Traits::RangeFieldType;
                        using size_type = typename LFSV::Traits::SizeType;
                        const unsigned int npe = lfsv_s.size();
                        typedef typename LFSV::Traits::FiniteElementType::
                            Traits::LocalBasisType::Traits::RangeType RT_V;

                        // Reference to the inside cell
                        const auto& cell_inside = ig.inside();

                        // Get geometry
                        auto geo = ig.geometry();

                        // Get geometry of intersection in local coordinates of cell_inside
                        auto geo_in_inside = ig.geometryInInside();

                        // evaluate boundary condition type
                        auto ref_el = referenceElement(geo_in_inside);
                        auto local_face_center = ref_el.position(0,0);
                        auto intersection = ig.intersection();

                        //If Dirichlet node return
                        if(param.isDirichlet(intersection, local_face_center)) return;

                        // loop over quadrature points and integrate normal flux
                        for (const auto& ip : quadratureRule(geo,intorder))
                        {
                            // position of quadrature point in local coordinates of element
                            auto local = geo_in_inside.global(ip.position());

                            //Evaluate Shape functions
                            std::vector<RT_V> phi(npe);
                            lfsv_s.finiteElement().localBasis().evaluateFunction(local,phi);

                            // evaluate flux boundary condition
                            auto j = param.j(cell_inside,local);

                            // integrate j
                            auto factor = ip.weight()*geo.integrationElement(ip.position());
                            for (size_type i=0; i<lfsu_s.size(); i++)
                                r_s.accumulate(lfsu_s,i,j*phi[i]*factor);
                        }
                    }


            private:
                int intorder;
                PARAM param;
                const GV gv;

        };

    }
}
#endif
