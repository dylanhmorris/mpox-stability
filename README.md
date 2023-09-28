# Stability of Monkeypox Virus in Body Fluids and Wastewater

Claude Kwe Yinda, Dylan H. Morris, Robert J. Fischer, Shane Gallogly, Zachary A. Weishampel, Julia R. Port, Trenton Bushmaker, Jonathan E. Schulz, Kyle Bibby, Neeltje van Doremalen, James O. Lloyd-Smith, and Vincent J. Munster

## Repository information
This repository accompanies the article "Stability of Monkeypox Virus in Body Fluids and Wastewater" (CK Yinda, DH Morris et al.). It provides code for reproducing Bayesian inference analyses from the paper and producing display figures.


## License and citation information
If you use the code or data provided here, please make sure to do so in light of the project [license](LICENSE) and please cite our work as below:

- C.K. Yinda, D.H. Morris, et al. Stability of Monkeypox Virus in Body Fluids and Wastewater. *Emerging Infectious Diseases* 29:10. October 2023. https://wwwnc.cdc.gov/eid/article/29/10/23-0824_article.

Bibtex record:
```
@article{yinda2023stabilitympox,
  title={Stability of Monkeypox Virus in Body Fluids and Wastewater},
  author={Yinda, Claude Kwe and
          Morris, Dylan H and
          Fischer, Robert J and
          Gallogly, Shane and
          Weishampel, Zachary A and
          Port, Julia R
          and Bushmaker, Trenton and
          Schulz, Jonathan E and
          Bibby, Kyle and
          van Doremalen, Neeltje and
          Lloyd-Smith, James O and
          Munster, Vincent J},
  journal={Emerging Infectious Diseases},
  volume={29},
  number={10},
  month={10},
  year={2023},
  url={https://wwwnc.cdc.gov/eid/article/29/10/23-0824_article}
}
```

## Article abstract 
An outbreak of human mpox infection in nonendemic countries appears to have been driven largely by transmission through body fluids or skin-to-skin contact during sexual activity. We evaluated the stability of monkeypox virus (MPXV) in different environments and specific body fluids and tested the effectiveness of decontamination methodologies. MPXV decayed faster at higher temperatures, and rates varied considerably depending on the medium in which virus was suspended, both in solution and on surfaces. More proteinaceous fluids supported greater persistence. Chlorination was an effective decontamination technique, but only at higher concentrations. Wastewater was more difficult to decontaminate than plain deionized water; testing for infectious MPXV could be a helpful addition to PCR-based wastewater surveillance when high levels of viral DNA are detected. Our findings suggest that, because virus stability is sufficient to support environmental MPXV transmission in healthcare settings, exposure and dose-response will be limiting factors for those transmission routes.

## Directories
- ``src``: all code, including Bayesian inference and figure generation:
- ``out``: mcmc and figure output files

# Reproducing analysis

A guide to reproducing the analysis from the paper follows. Code for the project should work on most standard macOS, Linux, and other Unix-like systems. It has not been tested in Windows. If you are on a Windows machine, you can try running the project within a [WSL2](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) environment containing Python.

You will also need a working $\TeX$ installation to render the text for the figures as they appear in the paper. If you do not have $\TeX$, you can either:
1. Install [TeXLive](https://tug.org/texlive/) (or another $\TeX$ distribution)
2. Turn off $\TeX$-based rendering by setting ``mpl.rcParams['text.usetex'] = False`` in this project's `src/plotting.py` file.

## Getting the code
First download the code. The recommended way is to ``git clone`` our Github repository from the command line:

    git clone https://github.com/dylanhmorris/mpox-stability.git

Downloading it manually via Github's download button should also work.

## Basic software requirements

The analysis can be auto-run from the project `Makefile`, but you may need to install some external dependencies first. In the first instance, you'll need a working installation of Python 3 (tested on Python 3.11) with the package manager `pip` and a working installation of Gnu Make or similar. Verify that you do by typing `which make` and `which pip` at the command line. 

## Virtual environments
If you would like to isolate this project's required dependencies from the rest of your system Python 3 installation, you can use a Python [virtual environment](https://docs.python.org/3/library/venv.html). 

With an up-to-date Python installation, you can create one by running the following command in the top-level project directory.

```
python3 -m venv .
```

Then activate it by running the following command, also from the top-level project directory.
```
source bin/activate
```

Note that if you close and reopen your Terminal window, you may need to reactivate that virtual environment by again running `source bin/activate`.

## Python packages
A few external python packages need to be installed. You can do so by typing the following from the top-level project directory.

    make install
    
or 

    pip install -r requirements.txt

Most of these packages are installed from the [Python Package Index](https://pypi.org/) except for the packages [`Pyter`](https://github.com/dylanhmorris/pyter) and [`Grizzlyplot`](https://github.com/dylanhmorris/grizzlyplot), which are pre-release Python packages developed by project co-author Dylan H. Morris, and which must be installed from Github. Those installs are linked to a specific git commit (i.e. version), with the goal of making it less likely that future changes to those packages make it difficult to reproduce the analysis here.

## Running analyses

The simplest approach is simply to type ``make`` at the command line, which should produce a full set of figures and results.

If you want to do things piecewise, typing ``make <filename>`` for any of the files present in the complete repository uploaded here should also work.

Some shortcuts are available:

- `make data` cleans raw data to produce cleaned data
- `make chains` produces all Markov Chain Monte Carlo output ("MCMC chains")
- `make figures` produces all figures
- `make tables` produces all tables.
- `make clean` removes all generated files, including even cleaned data, leaving only source code (though it does not uninstall packages)

## Note
While pseudorandom number generator seeds are set for reproducibility, the code has been updated to work with the latest versions several external Python packages, including Numpyro and Polars, since the figures and tables in the main text were generated, so numerical results may not be exactly identical.
