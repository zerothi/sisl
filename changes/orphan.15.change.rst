Removed possibility of doing Hk of integer datatypes

It increased compilation times significantly, with little gain.
Use float32/64 or complex64/128.
