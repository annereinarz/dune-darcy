add_executable("dune-darcy" dune-darcy.cc)
target_link_dune_default_libraries("dune-darcy")
dune_symlink_to_source_files(FILES configuration.ini)
dune_symlink_to_source_files(FILES grids)
