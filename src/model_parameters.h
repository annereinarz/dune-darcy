/* *************************************************** *
 * 
 *   User defined parameters for model problem
 *
 *  $Author: em459@BATH.AC.UK $
 *  $Revision: 32 $
 *  $Date: 2011-10-06 15:15:08 +0100 (Thu, 06 Oct 2011) $
 *
 * *************************************************** */

#ifndef __MODEL_PARAMETERS_H
#define __MODEL_PARAMETERS_H __MODEL_PARAMETERS_H

#ifdef ENABLE_MPI
#include<mpi.h>
#endif
#include<string.h>
#include<fstream>
#include<math.h>
#include<dune/common/parallel/mpihelper.hh>

using namespace std;

/* *************************************************** *
 * 
 *   Recursive structure for model parameters
 *
 * *************************************************** */

struct BoomerAMGParameters {
  int coarsentype;
  int interptype;
  int pmaxelmts;
  int aggnumlevels;
  int relaxtype;
  int relaxorder;
  double strongthreshold;
  int printlevel;
  int maxlevel;
  int coarsesolver;
  int ncoarserelax;
};

struct HypreParameters {
  BoomerAMGParameters boomeramg;
};

#endif
