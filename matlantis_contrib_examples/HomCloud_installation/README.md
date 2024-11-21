# Persistent homology analysis using HomCloud.
20240903 A. Nagoya, Preferred Computational Chemistry, Inc.

## HomCloud
**Persistent homology (PH)** is a tool from topological data analysis. PH is used to characterize the complex structural features of point clouds withoud clear periodicity, such as the atomic positions of amorphous solids, porous materials and polymers.
In this example, `HomCloud`, an python package for PH calculations, will be installed on Matlantis. Please refere the detailed description on the web site.
https://homcloud.dev/index.en.html
## References
* https://homcloud.dev/index.en.html
* I. Obayashi, J. Phys. Soc. Jpn. 91, 091013 (2022) https://journals.jps.jp/doi/10.7566/JPSJ.91.091013

## How to execute the examples.

Before installing `HomCloud`, please download and install the necessary packages `GNU MPFR` and `CGAL`.

1. Download the latest source archive `mpfr-4.2.X.tar.xz` of `GNU MPFR` from [the official website](https://www.mpfr.org/) and upload to your home directory on Matlantis.

2. Run the example notebook. It will download the source archive `CGAL-v5.6.X` via *wget* and build `GNU MPFR` and `CGAL`, then install `HomCloud` via *pip*.

GNU MPFR https://www.mpfr.org/ \
Dlownload: https://www.mpfr.org/mpfr-current/#download

CGAL https://www.cgal.org \
Github repositry: https://github.com/CGAL/cgal

## LICENSE
HomCloud is distributed under GPL 3 or any later version. For details, please see the LICENSE (https://homcloud.dev/download/LICENSE) oin the developer's homepage.
