# chebtest
Basic nanobench project for messing around with polynomial evaluation

```
mkdir -p build && cd build
cmake ..
make -j
taskset -c 0 ./chebtest
taskset -c 0 ./chebtest_ipo
```
