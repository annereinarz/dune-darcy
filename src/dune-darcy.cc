// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include<math.h>
#include<iostream>
#include<vector>
#include<map>
#include<string>
#include<sstream>
#include <mpi.h>

#include <config.h>

#include <dune/common/parametertree.hh>
Dune::ParameterTree configuration;

#include <dune/common/bitsetvector.hh>

#include <dune/grid/yaspgrid.hh> // Checked Inclusion
#include <dune/grid/common/gridview.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/io/file/vtk.hh>
#include <dune/grid/common/gridfactory.hh>
#include <dune/grid/common/gridinfo.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/io.hh>
#include <dune/istl/matrixmarket.hh>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parallel/collectivecommunication.hh>

#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/common/timer.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/grid/yaspgrid/partitioning.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/io.hh>
#include <dune/istl/superlu.hh>

#include <dune/pdelab/newton/newton.hh>
#include <dune/pdelab/finiteelementmap/p0fem.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/localfunctions/lagrange/qk.hh>
#include <dune/pdelab/finiteelementmap/rannacherturekfem.hh>
#include <dune/pdelab/constraints/common/constraints.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/vectorgridfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#include <dune/pdelab/gridfunctionspace/genericdatahandle.hh>
#include <dune/pdelab/gridfunctionspace/interpolate.hh>
#include <dune/pdelab/gridfunctionspace/vtk.hh>
#include <dune/pdelab/gridfunctionspace/subspace.hh>
#include <dune/pdelab/common/function.hh>
#include <dune/pdelab/common/vtkexport.hh>
#include <dune/pdelab/instationary/onestep.hh>
#include <dune/pdelab/common/instationaryfilenamehelper.hh>
#include <dune/pdelab/instationary/implicitonestep.hh>
#include <dune/pdelab/instationary/explicitonestep.hh>
#include <dune/pdelab/backend/istl.hh>
#include <dune/pdelab/backend/istl/bcrsmatrixbackend.hh>
#include <dune/pdelab/stationary/linearproblem.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <dune/pdelab/gridoperator/onestep.hh>
#include <dune/pdelab/instationary/onestepparameter.hh>
#include <dune/grid/geometrygrid/grid.hh>
#include <dune/common/parametertree.hh>
#include <dune/grid/io/file/gmshreader.hh>


#if HAVE_UG
#include <dune/grid/uggrid/uggridfactory.hh>
#endif

#include "hypreinterface.hh"
#include "problem_definition.hh"
#include "localoperator_darcy.hh"


//Start Main
int main(int argc, char** argv)
{
  try{

   // Maybe initialize MPI
   Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
   Dune::MPIHelper::MPICommunicator mycomm = helper.getCommunicator();

   //Read ini file
   Dune::ParameterTreeParser parser;
   parser.readINITree("configuration.ini",configuration);

   const int dim = 2;
   //Read in gmsh file
   typedef Dune::UGGrid<dim> GRID;
   GRID grid;
   Dune::GridFactory<GRID> factory(&grid);
   Dune::GmshReader<GRID>::read(factory,configuration.get<std::string>("mshName","grids/2dsquare.msh"));
   factory.createGrid();
   grid.loadBalance();

   //Create gridview
   typedef Dune::UGGrid<dim>::LeafGridView GV;
   const GV& gv=grid.leafGridView();

   //Set up boundary conditions
   typedef ConvectionDiffusionProblem<GV,double> Param;
   Param param;
   Dune::PDELab::BCTypeParam_CD<Param> bctype(gv,param);
   typedef Dune::PDELab::DirichletBoundaryCondition_CD<Param> G;
   G g(gv,param);

   // Construct grid function spaces
   typedef Dune::PDELab::NonoverlappingConformingDirichletConstraints<GV> CON;
   CON con(gv);

   //Set up FEM space
   const int element_order = 1; // Element order 1 - linear, 2 - quadratic
   const int dofel = 3;    //dofel depend on dim and element_order!
   typedef Dune::PDELab::PkLocalFiniteElementMap<GV,GV::Grid::ctype,double,element_order> FEM;
   FEM fem(gv);

   //Vector backend
   typedef Dune::PDELab::istl::VectorBackend<Dune::PDELab::istl::Blocking::none> VectorBackend;

   //Grid function space
   typedef Dune::PDELab::GridFunctionSpace<GV, FEM, CON, VectorBackend> GFS;
   GFS gfs(gv,fem,con); gfs.name("U");
   con.compute_ghosts(gfs);

   //Compute constrained space
   typedef typename GFS::template ConstraintsContainer<double>::Type C;
   C cg;
   Dune::PDELab::constraints( bctype, gfs, cg );

   //	Construct Linear Operator on FEM Space
   typedef Dune::PDELab::darcy<GV, Param, dofel> LOP;
   LOP lop(gv,param);

   //Assemble matrix using the local operator
   typedef Dune::PDELab::istl::BCRSMatrixBackend<> MBE;
   typedef Dune::PDELab::GridOperator<GFS,GFS,LOP,MBE,double,double,double,C,C> GO;
   auto go = GO(gfs,cg,gfs,cg,lop,MBE(30));

   // Make coefficent vector and initialize it from a function
   typedef Dune::PDELab::Backend::Vector<GFS,double> V;
   V x0(gfs,0.0);

   //Set up solver and solve linear system
   if(configuration.get<bool>("hypre")==false){
   typedef Dune::PDELab::ISTLBackend_NOVLP_CG_AMG_SSOR<GO> NOVLP_AMG;
   NOVLP_AMG ls(go,1000,1);
   Dune::PDELab::StationaryLinearProblemSolver<GO,NOVLP_AMG,V> slp(go,ls,x0,1e-6);
   slp.apply();
   }
//Hypre
   if(configuration.get<bool>("hypre")){
  HypreParameters hypre_param;
        /*hypre_param.boomeramg.coarsentype = ;
        hypre_param.boomeramg.interptype;
        hypre_param.boomeramg.pmaxelmts;
        hypre_param.boomeramg.aggnumlevels;
        hypre_param.boomeramg.relaxtype;
        hypre_param.boomeramg.relaxorder;
        hypre_param.boomeramg.strongthreshold;
        hypre_param.boomeramg.printlevel;
        hypre_param.boomeramg.maxlevel;
        hypre_param.boomeramg.coarsesolver;
        hypre_param.boomeramg.ncoarserelax;*/
        HypreSolver<GO> solver(go,x0, 1e-6 ,hypre_param);
        solver.solve(x0);
  }

   // graphics
   typedef Dune::PDELab::DiscreteGridFunction<GFS,V> DGF;
   DGF xdgf(gfs,x0);
   Dune::VTKWriter<GV> vtkwriter(gv,Dune::VTK::conforming);
   vtkwriter.addVertexData(std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<DGF> >(xdgf,"solution"));
   vtkwriter.write("TestModel",Dune::VTK::appendedraw);

   return 0;
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}



