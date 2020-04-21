# DEAP_MNIST_CLS

Experiment for data split with GA(deap).
For the dataset, mnist is used.

## Requirements
- [Docker](https://www.docker.com/) >= 19.03
- [GNU Make](https://www.gnu.org/software/make/)
- [nvidia-drivers](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) (Only for GPU)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (Only for GPU)

## Installation
### Clone repository
```bash
$ git clone https://github.com/TerauchiTonnnura/DEAP_MNIST_CLS
$ cd DEAP_MNIST_CLS
```

### Build image
```bash
$ make build
```

**NOTE:** <br>
If you want use GPUs, install [nvidia-drivers](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) first.

## How to Use

<hr class="footnotes-sep">
<section class="footnotes">
    <ol class="footnotes-list">
        <li id="fn1"  class="footnote-item">
            <p>
                It cannot handle packages which have a specific version for GPU environment like tensorflow.
                Therefore they are installed in Dockerfiles.
                <a href="#fnref1" class="footnote-backref">↩</a>
            </p>
        </li>
    </ol>
</section>
