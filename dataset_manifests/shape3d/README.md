# Shape3D data manifest

The official `3dshapes.h5` and the prepared image arrays are external data.
The existing balanced 20,000-sample preparation procedure will be migrated
without changing its selection, preprocessing, or seed.

`attribution_5k_indices.npy` is a release manifest, not a raw dataset file. It
contains the fixed seed-42 candidate set used by the experiment. Its expected
SHA256 is recorded in `attribution_scope.json`.
