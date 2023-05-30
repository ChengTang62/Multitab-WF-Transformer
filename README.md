# Multi-tab WF Attack datasets
Public datasets for evaluating multi-tab website fingerprinting attacks.

If you want to cite the datasets library, you can use our [paper](https://www.computer.org/csdl/proceedings-article/sp/2023/933600b005/1NrbYpaG652).
```bibtex
@INPROCEEDINGS {multitab-wf-datasets,
author = {X. Deng and Q. Yin and Z. Liu and X. Zhao and Q. Li and M. Xu and K. Xu and J. Wu},
booktitle = {2023 IEEE Symposium on Security and Privacy (SP)},
title = {Robust Multi-tab Website Fingerprinting Attacks in the Wild},
year = {2023},
volume = {},
issn = {},
pages = {1005-1022},
abstract = {},
keywords = {},
doi = {10.1109/SP46215.2023.00122},
url = {https://doi.ieeecomputersociety.org/10.1109/SP46215.2023.00122},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {may}
}
```

## Datasets

You can download the dataset via the [link](https://drive.google.com/file/d/1akeBzeGLfnzgmD0Qt196WshwgbsYMnnS/view?usp=sharing).

All datasets can be loaded by numpy.

```python
import numpy as np

inpath = "example.npz"
data = np.load(inpath)
dir_array = data["direction"]  # Sequence of packet direction
time_array = data["time"] # Sequence of packet timestamps
label = data["label"]  # labels
```


Note: We have further improved the quality of the dataset. Due to limitations in server hardware performance, collecting data with a multi-tab setup may result in inaccessible web pages. By capturing screenshots of the webpage loading conditions using xvfbwrapper, we filtered traces from unsuccessful access, thereby enhancing the dataset's reliability.