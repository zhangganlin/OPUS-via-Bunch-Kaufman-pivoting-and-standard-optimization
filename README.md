# Optimization by Particle swarm Using Surrogates via Bunch-Kaufman pivoting and standard optimization
---
Course project of Advancecd System Lab Spring 2022 in ETHZ. 

Focus on speeding up black box optimization algorithm OPUS from pape "Particle swarm with radial basis function surrogates for expensive black-box optimization" by Rommel G. Regis.

Details about the optimization we did can be found in our project [report](22_report.pdf)

### Author: [Ganlin Zhang](https://github.com/zhangganlin), [Deheng Zhang](https://github.com/dehezhang2), [Junpeng Gao](https://github.com/JunpengGao233), [Yu Hong](https://github.com/YUYHY)

### Dependency:
* Ceres-solver [http://ceres-solver.org/](http://ceres-solver.org/installation.html)
* Eigen [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/index.php?title=Main_Page) (For baseline version only)
  
### Demo
```bash
mkdir build && cd build
cmake ..
make
./demo
```
### Blocking Bunch-Kaufman Pivoting
In this project, we use Bunch-Kaufman Pivoting to solve linear systems. To find the suitable blocking size, we also provide an automatic tool:
```bash
cd build
./test_block_bk
```
It may take hours to find proper blocking size, but on each machine, it only need to be run once.




