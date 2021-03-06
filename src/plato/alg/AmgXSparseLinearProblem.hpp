//
//  AmgXSparseLinearProblem.hpp
//  
//
//  Created by Roberts, Nathan V on 8/8/17.
//
//
#ifndef LGR_AMGX_SPARSE_LINEAR_PROBLEM_HPP
#define LGR_AMGX_SPARSE_LINEAR_PROBLEM_HPP

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#endif

#include "plato/alg/CrsLinearProblem.hpp"
#include "plato/alg/AmgXConfigs.hpp"
#include <plato/PlatoTypes.hpp>

#include <amgx_c.h>
#include <sstream>
#include <fstream>

#include <cassert>

namespace Plato {
  template<class Ordinal, int BlockSize=1>
  class AmgXSparseLinearProblem : public CrsLinearProblem<Ordinal>
  {

  public:
    typedef Kokkos::View<Scalar*, MemSpace>                        Vector;
  private:
    typedef int RowMapEntryType;
    typedef CrsMatrix<Ordinal>  Matrix;
    
    typedef Ordinal LocalOrdinal;
    typedef Ordinal GlobalOrdinal;
    
    AMGX_matrix_handle    _matrix;
    AMGX_vector_handle    _rhs;
    AMGX_vector_handle    _lhs;
    AMGX_resources_handle _rsrc;
    AMGX_solver_handle    _solver;
    AMGX_config_handle    _config;
    
    bool _haveInitialized = false;

    Vector _x; // will want to copy here (from _lhs) in solve()...

  public:
    static void initializeAMGX()
    {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_initialize_plugins());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }
    
    AmgXSparseLinearProblem(const Matrix A, Vector x, const Vector b, std::string const& solverConfigString = configurationString("pcg_noprec"))
                          : CrsLinearProblem<Ordinal>(A,x,b)
    {
      check_inputs(A, x, b);

      initializeAMGX();
      AMGX_config_create(&_config, solverConfigString.c_str());
      // everything currently assumes exactly one MPI rank.
      MPI_Comm mpi_comm = MPI_COMM_SELF;
      int ndevices = 1;
      int devices[1];
      //it is critical to specify the current device, which is not always zero
      cudaGetDevice(&devices[0]);
      AMGX_resources_create(
          &_rsrc, _config, &mpi_comm, ndevices, devices);
      AMGX_matrix_create(&_matrix, _rsrc, AMGX_mode_dDDI);
      AMGX_vector_create(&_rhs,    _rsrc, AMGX_mode_dDDI);
      AMGX_vector_create(&_lhs,    _rsrc, AMGX_mode_dDDI);
      
      _x = x;
      Ordinal N = x.size();
      Ordinal nnz = A.columnIndices().size();
      
      AMGX_solver_create(&_solver, _rsrc, AMGX_mode_dDDI, _config);
      
      // This seems to do the right thing whether the data is on device or host. In our case it is on the device.
      const int *row_ptrs = A.rowMap().data();
      const int *col_indices = A.columnIndices().data();
      const void *data = A.entries().data();
      const void *diag_data = nullptr; // no exterior diagonal
      AMGX_matrix_upload_all(_matrix, N/BlockSize, nnz, BlockSize, BlockSize, row_ptrs, col_indices, data, diag_data);
      
      setRHS(b);
      setInitialGuess(x);
    }
    
    void initializePreconditioner()
    {
      Kokkos::Profiling::pushRegion("AMGX_solver_setup");
      AMGX_solver_setup(_solver, _matrix);
      Kokkos::Profiling::popRegion();
    }
    
    void initializeSolver() // TODO: add mechanism for setting options
    {
      initializePreconditioner();
    }
    
    void setInitialGuess(const Vector x)
    {
      AMGX_vector_upload(_lhs, x.size()/BlockSize, BlockSize, x.data());
    }
    
    void setMaxIters(int maxCGIters)
    {
      // NOTE: this does not work; we're getting the format wrong, somehow
      std::ostringstream cfgStr;
      cfgStr << "config_version=2" << std::endl;
      cfgStr << "solver:max_iters=" << maxCGIters << std::endl;
      AMGX_config_add_parameters(&_config, cfgStr.str().c_str());
    }

    void setRHS(const Vector b)
    {
      AMGX_vector_upload(_rhs, b.size()/BlockSize, BlockSize, b.data());
    }

    void setMatrix(const Matrix & aMatrix, const Ordinal & aNumEquations)
    {
        const void *tData = aMatrix.entries().data();
        const void *tDiagData = nullptr; // no exterior diagonal
        const int *tRowPtrs = aMatrix.rowMap().data();
        const int *tColIndices = aMatrix.columnIndices().data();
        const Ordinal tNumNonZeros = aMatrix.columnIndices().size();
        AMGX_matrix_upload_all(_matrix, aNumEquations/BlockSize, tNumNonZeros, BlockSize, BlockSize, tRowPtrs, tColIndices, tData, tDiagData);
    }
    
    void setTolerance(Plato::Scalar tol)
    {
      // NOTE: this does not work; we're getting the format wrong, somehow
      std::ostringstream cfgStr;
      cfgStr << "config_version=2" << std::endl;
      cfgStr << "solver:tolerance=" << tol << std::endl;
      AMGX_config_add_parameters(&_config, cfgStr.str().c_str());
    }

    int solve()
    {
      using namespace std;

      if (!_haveInitialized)
      {
        initializeSolver();
        _haveInitialized = true;
      }
      int err = cudaDeviceSynchronize();
      assert(err == cudaSuccess);
      Kokkos::Profiling::pushRegion("AMGX_solver_solve");
      auto solverErr = AMGX_solver_solve(_solver, _rhs, _lhs);
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("AMGX_vector_download");
      AMGX_vector_download(_lhs, _x.data());
      Kokkos::Profiling::popRegion();
      return solverErr;
    }

    ~AmgXSparseLinearProblem()
    {
      AMGX_solver_destroy    (_solver);
      AMGX_matrix_destroy    (_matrix);
      AMGX_vector_destroy    (_rhs);
      AMGX_vector_destroy    (_lhs);
      AMGX_resources_destroy (_rsrc);
      
      AMGX_SAFE_CALL(AMGX_config_destroy    (_config));

      AMGX_SAFE_CALL(AMGX_finalize_plugins());
      AMGX_SAFE_CALL(AMGX_finalize());
    }
    static std::string getConfigString(int aMaxIters = 1000)
    {
      std::string configString;
      
      Plato::Scalar tol = 1e-12;
      int maxIters = aMaxIters;
      
      std::ifstream infile;
      infile.open("amgx.json", std::ifstream::in);
      if(infile){
        std::string line;
        std::stringstream config;
        while (std::getline(infile, line)){
          std::istringstream iss(line);
          config << iss.str();
        }
        configString = config.str();
      } else {
        configString = configurationString("pcg_noprec",tol,maxIters);
      }
    
      return configString;
    }

    static void check_inputs(const Matrix A, Vector x, const Vector b) {
      auto ndofs = int(x.extent(0));
      assert(int(b.extent(0)) == ndofs);
      assert(ndofs % BlockSize == 0);
      auto nblocks = ndofs / BlockSize;
      auto row_map = A.rowMap();
      assert(int(row_map.extent(0)) == nblocks + 1);
      auto col_inds = A.columnIndices();
      auto nnz = int(col_inds.extent(0));
      assert(int(A.entries().extent(0)) == nnz * BlockSize * BlockSize);
      assert(cudaSuccess == cudaDeviceSynchronize());
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, nblocks), KOKKOS_LAMBDA(int i) {
        auto begin = row_map(i);
        assert(0 <= begin);
        auto end = row_map(i + 1);
        assert(begin <= end);
        if (i == nblocks - 1) assert(end == nnz);
        else assert(end < nnz);
        for (int ij = begin; ij < end; ++ij) {
          auto j = col_inds(ij);
          assert(0 <= j);
          assert(j < nblocks);
        }
      }, "check_inputs");
      assert(cudaSuccess == cudaDeviceSynchronize());
    }

  };
}

#endif /* AmgXSparseLinearProblem_h */
