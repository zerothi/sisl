Consolidated the element-deletion + pointer-update idiom in `SparseCSR`

The three copies of the element deletion logic
now share a single ``SparseCSR._delete_stored`` helper.
This is a pure internal refactor with no change in behavior.
