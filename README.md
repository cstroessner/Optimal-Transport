Code to reproduce the figures in "Low-rank tensor approximations for solving multi-marginal optimal transport problems" by C. Strössner and D. Kressner.
(SIAM J. Imaging Sci. 2023 16:1, 69-191 https://doi.org/10.1137/22M1478355, Arxiv: https://arxiv.org/abs/2202.07340)

To run the experiments you need to additionally download:
- the TT toolbox (https://github.com/oseledets/TT-Toolbox) for the comparison to the TT Approximation
- the Val images 2017 (http://images.cocodataset.org/zips/val2017.zip) of the COCO data set (https://cocodataset.org/#home) for the color transfer example

You need to manually add the TT toolbox to the MATLAB path and run its setup script. The Folder with the images needs to be added to the MATLAB path manually.

The script PlotSection61 reproduces all plots in Section 6.1. of the article. The script PlotColorTransfer reproduces all results for the color transfer example in the arxiv version. The scripts PlotSchoedingerBridges and PlotColorBarycenters reproduce the figures in the revised version.
