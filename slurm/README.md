# Cluster launchers

Machine-specific launchers live here, outside experiment folders. Launchers
must receive dataset and output roots through `PINT_DATA_ROOT` and
`PINT_OUTPUT_ROOT`; they must not contain user-specific absolute paths.
