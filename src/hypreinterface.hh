/* *************************************************** *
 * *************************************************** *
 *
 *    Hypre solver interface
 *
 *  $Author: em459@BATH.AC.UK $
 *  $Revision: 319 $
 *  $Date: 2012-07-27 23:57:57 +0100 (Fri, 27 Jul 2012) $
 *
 * *************************************************** *
 * *************************************************** */
#ifndef HYPREINTERFACE_HH
#define HYPREINTERFACE_HH HYPREINTERFACE_HH

#include<vector>
#include<string>
#include<dune/common/parallel/collectivecommunication.hh>
#include<dune/common/parallel/mpihelper.hh>
#include<dune/pdelab/gridoperator/gridoperator.hh>
#include<dune/grid/common/indexidset.hh>
#include<dune/grid/common/gridview.hh>
#include<dune/istl/bcrsmatrix.hh>
#include<dune/istl/solvers.hh>
#include"../../../hypre-2.11.2/src/hypre/include/HYPRE.h"
#include"../../../hypre-2.11.2/src/hypre/include/HYPRE_IJ_mv.h"
#include"../../../hypre-2.11.2/src/hypre/include/HYPRE_parcsr_ls.h"
#include"../../../hypre-2.11.2/src/hypre/include/HYPRE_krylov.h"
#include"model_parameters.h"

/* *************************************************** *
 * Solver class
 * *************************************************** */
template <class GO>
class HypreSolver : public Dune::PDELab::LinearResultStorage {
 public:
  // Extracted types
  typedef typename GO::Traits::TrialGridFunctionSpace::Traits::GridViewType GV; // Gridview
  typedef typename GO::Traits::Domain U;
  typedef typename GO::Jacobian Matrix;
  /* Constructor */
  HypreSolver (const GO& go_, U& u_, 
               const double tolerance_,
               const HypreParameters& hypre_param_) :
    go(go_), u(u_), b(u_), 
    gv(go_.trialGridFunctionSpace().gridView()), 
    tolerance(tolerance_),
    maxiter(100),
    ParCSRPCG_printlevel(3),
    hypre_param(hypre_param_) {
    // Build global index map
    BuildGlobalIndexMap();
    // Set up BoomerAMG preconditioner
    HYPRE_BoomerAMGCreate(&prec);
    HYPRE_BoomerAMGSetTol(prec,0);
    HYPRE_BoomerAMGSetMaxIter(prec,1);
    /*HYPRE_BoomerAMGSetInterpType(prec, hypre_param.boomeramg.interptype); 
    HYPRE_BoomerAMGSetPMaxElmts(prec, hypre_param.boomeramg.pmaxelmts);
    HYPRE_BoomerAMGSetCoarsenType(prec, hypre_param.boomeramg.coarsentype);
    HYPRE_BoomerAMGSetAggNumLevels(prec, hypre_param.boomeramg.aggnumlevels);
    HYPRE_BoomerAMGSetRelaxType(prec, hypre_param.boomeramg.relaxtype);
    HYPRE_BoomerAMGSetRelaxOrder(prec, hypre_param.boomeramg.relaxorder);
    HYPRE_BoomerAMGSetMaxLevels (prec, hypre_param.boomeramg.maxlevel);
    HYPRE_BoomerAMGSetCycleNumSweeps(prec, hypre_param.boomeramg.ncoarserelax,3); 
    HYPRE_BoomerAMGSetCycleRelaxType(prec, hypre_param.boomeramg.coarsesolver,3); 
    HYPRE_BoomerAMGSetPrintLevel(prec,hypre_param.boomeramg.printlevel); 
    HYPRE_BoomerAMGSetStrongThreshold(prec,hypre_param.boomeramg.strongthreshold);*/
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD,&solver);
    HYPRE_ParCSRPCGSetTol(solver,tolerance);
    HYPRE_ParCSRPCGSetMaxIter(solver,maxiter);
    HYPRE_ParCSRPCGSetPrintLevel(solver, ParCSRPCG_printlevel);
    HYPRE_PCGSetTwoNorm(solver,1);
    if (gv.comm().rank() == 0) {
      std::cout << " ParCSRPCG tolerance = "
                << tolerance << std::endl;
      std::cout << " ParCSRPCG maxiter = "
                << maxiter << std::endl;
    }
    HYPRE_ParCSRPCGSetPrecond(solver, 
                        HYPRE_BoomerAMGSolve,
                        HYPRE_BoomerAMGSetup,
                        prec);
  }

  /* Destructor */
  ~HypreSolver() {
    HYPRE_BoomerAMGDestroy(prec);
    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_IJVectorDestroy(b_hypre);
    HYPRE_IJMatrixDestroy(a_hypre);
  }

  // Solve for a given RHS
  void solve(U& u) {
    Dune::Timer watch;
    double timing;
    watch.reset();
    Matrix a(go);
    timing = watch.elapsed();
    timing = gv.comm().max(timing);
    if (gv.comm().rank()==0)
          std::cout << "=== matrix setup (max) " << timing << " s" << std::endl;
    watch.reset();
    go.jacobian(u,a);
    timing = watch.elapsed();
    timing = gv.comm().max(timing);
    if (gv.comm().rank()==0)
          std::cout << "=== matrix assembly (max) " << timing << " s" << std::endl;
    watch.reset();
    go.residual(u,b);
    timing = watch.elapsed();
    b *= -1.0;
    timing = gv.comm().max(timing);
    if (gv.comm().rank()==0)
          std::cout << "=== residual assembly (max) " << timing << " s" << std::endl;
    AssembleHypreMatrix(a,a_hypre);
    AssembleHypreVector(b,b_hypre); 
    HYPRE_IJVector u_hypre;
    AssembleHypreVector(u,u_hypre); 
    // Extract ParCSR data
    HYPRE_ParCSRMatrix a_parcsr;
    HYPRE_ParVector b_parcsr;
    HYPRE_ParVector u_parcsr;
    HYPRE_IJMatrixGetObject(a_hypre,(void **) &a_parcsr); 
    HYPRE_IJVectorGetObject(b_hypre,(void **) &b_parcsr); 
    HYPRE_IJVectorGetObject(u_hypre,(void **) &u_parcsr); 


    //HYPRE_IJMatrixPrint(a_hypre, "matrix-out.dat");
    watch.reset();
    HYPRE_ParCSRPCGSetup(solver, a_parcsr, b_parcsr, u_parcsr);
    double timeBoomerAMGSetup = watch.elapsed();
    timeBoomerAMGSetup = gv.comm().max(timeBoomerAMGSetup);
    if (gv.comm().rank() == 0)
      std::cout << " BoomerAMG Setup time (max) : " << timeBoomerAMGSetup << " s " << std::endl;
    watch.reset(); 
    HYPRE_ParCSRPCGSolve(solver, a_parcsr, b_parcsr,u_parcsr);
    double timeBoomerAMGSolve = watch.elapsed();
    timeBoomerAMGSolve = gv.comm().max(timeBoomerAMGSolve);
    if (gv.comm().rank() == 0)
      std::cout << " BoomerAMG Solve time (max) : " << timeBoomerAMGSolve << " s " << std::endl; 
    ReadHypreVector(u_hypre,u);
    HYPRE_IJVectorDestroy(u_hypre);
    HYPRE_Int iterations;
    double resreduction;
    HYPRE_ParCSRPCGGetNumIterations(solver,&iterations);
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver,&resreduction);
    res.converged  = (resreduction <= tolerance);
    res.iterations = iterations;
    res.reduction = resreduction;
    res.elapsed    = timeBoomerAMGSolve;
    res.conv_rate  = pow(resreduction,1./iterations);
  }

  /* Read solver tolerance */
  double Tolerance() const { return tolerance; }

 private:
  
/* *************************************************** *
 * Subclass for halo exchange. This is needed for 
 * the construction of a global mapping between local
 * indices and a global consecutive index.
 * *************************************************** */
  typedef typename std::vector<HYPRE_Int> GlobalIndexMap;
  template <class IndexSet, class V>
  class VectorExchange :
    public Dune::CommDataHandleIF<VectorExchange<IndexSet,V>,
                                  typename V::value_type> {
   public:
    typedef typename V::value_type DataType; // Data type
    bool contains (int dim, int codim) const { return (codim == 0); }
    bool fixedsize (int dim, int codim) const { return true; }
    template <class EntityType>
    // Extract size (only one data item per element)
    size_t size(EntityType& e) const { return 1; }
    // Write data to message buffer
    template <class MessageBuffer, class EntityType>
    void gather (MessageBuffer& buff, const EntityType& e) const {
      buff.write(c_[indexset_.index(e)]);
    }
    // Read data from message buffer
    template <class MessageBuffer, class EntityType>
    void scatter (MessageBuffer& buff, const EntityType& e, size_t n) {
      DataType x;
      buff.read(x);
      c_[indexset_.index(e)] = x;
    }
   // Constructor
   VectorExchange(const IndexSet& indexset, V& c) :
      indexset_(indexset), c_(c) {}
   private:
    const IndexSet& indexset_; // map entity -> index set
    V& c_; // Data container
  };

/* *************************************************** *
 * Create a map which of type GlobalIndexMap which 
 * returns a consecutive global index for each local
 * index in the indexset of the underlying gridview. 
 * *************************************************** */
  void BuildGlobalIndexMap() {
    // Loop over all interior entities of codim 0 and
    // order them consecutively, i.e. m[idx] = i where
    // i = 0,...,n_I
    typedef typename GV::template Codim<0>::template Partition<Dune::Interior_Partition>::Iterator EIterator;
    EIterator ibegin = gv.template begin<0,Dune::Interior_Partition>();
    EIterator iend = gv.template end<0,Dune::Interior_Partition>();
    m.resize(gv.indexSet().size(0));
    HYPRE_Int nI = 0;
    for (EIterator it=ibegin;it!=iend;++it,++nI) {
      size_t idx = gv.indexSet().index(*it);
      m[idx] = nI;
    }
    // Communicate the size n_I of the set of local interior elements
    // and calculate the first index as 
    //    firstIndex = sum_{p'<p} |n_I^{(p)'}|
    // then the unique consecutive global index is i+firstIndex
    size_t commSize = gv.comm().size(); 
    size_t commRank = gv.comm().rank(); 
    HYPRE_Int *S = new HYPRE_Int[commSize];
    gv.comm().allgather(&nI,1,S);
    firstIndex=0;
    for (size_t rank=0;rank<commRank;++rank) {
      firstIndex += S[rank];
    }
    lastIndex = firstIndex+(nI-1);
    delete [] S;
    for (EIterator it=ibegin;it!=iend;++it,++nI) {
      int idx = gv.indexSet().index(*it);
      m[idx] += firstIndex;
    }
    // Communicate local indices on halo elements
    VectorExchange<typename GV::IndexSet,GlobalIndexMap>
      haloexchanger(gv.indexSet(),m);
    gv.template communicate<VectorExchange<typename GV::IndexSet,GlobalIndexMap>>(haloexchanger,
      Dune::InteriorBorder_All_Interface,
      Dune::ForwardCommunication);
}
/* *************************************************** *
 * Assemble hypre matrix given matrix a
 * *************************************************** */
  void AssembleHypreMatrix(const Matrix& a, 
                           HYPRE_IJMatrix& hypre_matrix) {
    HYPRE_Int ilower, iupper;
    HYPRE_Int jlower, jupper;
    ilower = 0;
    iupper = a.M();
    jlower = 0;
    jupper = a.N();
    HYPRE_IJMatrixCreate(gv.comm(), 
                         ilower, iupper,
                         jlower, jupper, 
                         &hypre_matrix);
    HYPRE_IJMatrixSetObjectType(hypre_matrix, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(hypre_matrix);
    using Dune::PDELab::Backend::native;
    //printmatrix(std::cout, native(a), "b", "b");
    int nrows=1;
    HYPRE_Int ncol=1;
    for (int i_row = 0; i_row < a.M(); i_row++) {
        for (int i_col = 0; i_col < a.N(); i_col++) {
            if(native(a).exists(i_row,i_col)){
                HYPRE_Int row_idx = i_row;
                HYPRE_Int col_idx = i_col;
                double value = native(a)[i_row][i_col];
                HYPRE_IJMatrixSetValues(hypre_matrix,nrows,&ncol,&row_idx,&col_idx,&value);
            }
        }
    }
    HYPRE_IJMatrixAssemble(hypre_matrix);
  }
/* *************************************************** *
 * Assemble hypre vector given vector u and return 
 * *************************************************** */
  void AssembleHypreVector(const U& u, 
                           HYPRE_IJVector& u_hypre) {
    HYPRE_Int jlower = 0;
    HYPRE_Int jupper = u.N();
    HYPRE_IJVectorCreate(gv.comm(), jlower, jupper, &u_hypre);
    HYPRE_IJVectorSetObjectType(u_hypre, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(u_hypre);
    using Dune::PDELab::Backend::native;
    int nvalues=1;
    for (int i = 0; i < u.N(); i++) {
      HYPRE_Int indices = i;
      double values = native(u)[i];
      HYPRE_IJVectorSetValues(u_hypre, nvalues, &indices, &values); 
    }
    HYPRE_IJVectorAssemble(u_hypre);
  }

/* *************************************************** *
 * Extract from hypre vector u and return 
 * *************************************************** */
  void ReadHypreVector(const HYPRE_IJVector& u_hypre,
                       U& u) {
    HYPRE_Int jlower = 0;
    HYPRE_Int jupper = u.N();
    int nvalues=1;
    double values;
    using Dune::PDELab::Backend::native;
    for (int i = 0; i < u.N(); i++) {
      HYPRE_Int indices = i;
      HYPRE_IJVectorGetValues(u_hypre, nvalues, &indices, &values); 
      native(u)[i] = values;
    }
  }
/* *************************************************** *
 * Class data 
 * *************************************************** */
  GlobalIndexMap m;
  HYPRE_Int firstIndex; // first global interior index  
  HYPRE_Int lastIndex;  // last global interior index
  const GO& go;   // Gridoperator
  const GV& gv;   // gridview of trial grid function space
  U& u;           // Point where the Jacobian is evaluated
  U b;            // RHS
  HYPRE_IJMatrix a_hypre;  // Hypre matrix storage
  HYPRE_IJVector b_hypre;
  HYPRE_Solver prec;
  HYPRE_Solver solver;
  double tolerance; // ParCSRPCG Solver tolerance
  HYPRE_Int maxiter; // ParCSRPCG maxiter
  // BoomerAMG parameters
  const HypreParameters& hypre_param;
  int ParCSRPCG_printlevel;
};
#endif
