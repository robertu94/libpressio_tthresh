# `pressio_tthresh`

`pressio_tthresh` is a [LibPressio](https://github.com/robertu94/libpressio) compatible implementation of tthresh written by Robert Underwood.
The code borrows most of the original tthresh implementation, whose description can be found below.

## Multidimensional Compression Using the Tucker Tensor Decomposition

TThresh is an **open-source C++ implementation** written by [Rafael Ballester-Ripoll](http://www.ifi.uzh.ch/en/vmml/people/current-staff/ballester.html) (rballester@ifi.uzh.ch) of the compressor developed in [TTHRESH: Tensor Compression for Multidimensional Visual Data (R. Ballester-Ripoll, P. Lindstrom and R. Pajarola)](https://arxiv.org/pdf/1806.05952.pdf). It is intended for Cartesian grid data of **3 or more dimensions**, and leverages the higher-order singular value decomposition (HOSVD), a generalization of the SVD to 3 and more dimensions.

If you use TTHRESH for a scientific publication, please cite one or both of these papers:

- [*TTHRESH: Tensor Compression for Multidimensional Visual Data*](https://arxiv.org/abs/1806.05952):

```@article{BLP:19, Author = {Ballester-Ripoll, Rafael and Lindstrom, Peter and Pajarola, Renato}, Journal = {IEEE Transaction on Visualization and Computer Graphics}, Keywords = {visualization, data compression, volume rendering, higher-order decompositions, tensor approximation}, Note = {arXiv:1806.05952}, Title = {TTHRESH: Tensor Compression for Multidimensional Visual Data}, Volume = {to appear}, Year = {2019}}```
- [*Lossy Volume Compression Using Tucker Truncation and Thresholding*](http://www.ifi.uzh.ch/en/vmml/publications/lossycompression.html): ```@article{BP:15, year={2015}, issn={0178-2789}, journal={The Visual Computer}, title={Lossy volume compression using {T}ucker truncation and thresholding}, publisher={Springer Berlin Heidelberg}, keywords={Tensor approximation; Data compression; Higher-order decompositions; Tensor rank reduction; Multidimensional data encoding}, author={Ballester-Ripoll, Rafael and Pajarola, Renato}, pages={1-14}}
```

For more information on the Tucker transform and tensor-based volume compression, check out the [slides](http://www.ifi.uzh.ch/dam/jcr:00000000-73a0-83b8-ffff-ffffd48b8a42/tensorapproximation.pdf) of the authors of tthresh.

### Download

```bash
git clone https://github.com/rballester/tthresh.git
```

(or as a [zip file](https://github.com/rballester/tthresh/archive/master.zip)).

### Compilation

Use [CMake](https://cmake.org/) to generate an executable ```tthresh```:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Usage

Link to the library, and then use the libpressio interface.

### Acknowledgments

The original work has this acknowledgments:

This work was partially supported by the [UZH Forschungskredit "Candoc"](http://www.researchers.uzh.ch/en/funding/phd/fkcandoc.html), grant number FK-16-012. I also thank [Enrique G. Paredes](http://www.ifi.uzh.ch/en/vmml/people/current-staff/egparedes.html) for his help with CMake compilation issues.

### Why Tucker?

Tensor-based compression is non-local, in the sense that all compressed coefficients contribute to the reconstruction of each individual voxel (in contrast to e.g. wavelet transforms or JPEG for images, which uses a localized DCT transform). This can be computationally demanding but decorrelates the data at all spatial scales, which has several advantages:

- Very competitive **compression quality**
- Fine bit-rate **granularity**
- **Smooth degradation** at high compression (in particular, no blocking artifacts or temporal glitches)
- Ability to **downsample** in the compressed domain
