add_executable("dune-darcy" dune-darcy.cc)
target_link_dune_default_libraries("dune-darcy")
dune_symlink_to_source_files(FILES configuration.ini)
dune_symlink_to_source_files(FILES grids)
#link to hypre
include_directories("/home/anne/Desktop/DUNE-2.5.1/hypre-2.11.2/src/hypre/include")
LINK_DIRECTORIES(/home/anne/Desktop/DUNE-2.5.1/hypre-2.11.2/src/hypre/include/)
TARGET_LINK_LIBRARIES(dune-darcy /home/anne/Desktop/DUNE-2.5.1/hypre-2.11.2/src/hypre/lib/libHYPRE.a)
