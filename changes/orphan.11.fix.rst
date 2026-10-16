Fixed a user pool in `BrillouinZone.apply`

``bz.apply.renew(pool=pool)`` with a pool object raised a `NameError`. It now uses the
number of CPUs of the pool, and warns only when it cannot find that number.
