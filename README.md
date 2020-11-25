# /ru/ConceptNet
ConceptNet 5.7 (Russian part) extraction scripts + fast API object to access the relations. Note: a simple modification of the 
preprocessing script allows to build a queryable graph of **any other subset of ConceptNet**.


![Python 3x](https://img.shields.io/badge/python-3.x-blue.svg)
[![PyPI version][pypi_badge]][pypi_link]
[![Downloads](https://pepy.tech/badge/ruconceptnet)](https://pepy.tech/project/ruconceptnet)

[pypi_badge]: https://badge.fury.io/py/ruconceptnet.svg
[pypi_link]: https://pypi.python.org/pypi/ruconceptnet

### Installation

```
pip install ruconceptnet
```

### Usage

```
>>> from ruconceptnet import ConceptNet
>>> cn = ConceptNet()
>>> cn.get_targets("алкоголь")
[('этиловый_спирт', {'Synonym'}), ('спиртной_напиток', {'Synonym'}), ('алкогольный', {'RelatedTo'}), 
('алкоголик', {'RelatedTo'}), ('спирт', {'Synonym'}), ('алкоголизация', {'RelatedTo'})]

>>> cn.get_sources("йога")
[('йоги', {'FormOf'}), ('йогу', {'FormOf'}), ('йогический', {'RelatedTo'}), ('йогою', {'FormOf'}), 
('йогой', {'FormOf'}), ('йог', {'RelatedTo'}), ('йоге', {'FormOf'})]

>>> cn.check_pair("человек", "зверь")
(['DistinctFrom'], [])

>>> cn.check_pair("зверь", "человек")
([], ['DistinctFrom'])
```

### Preparations for customization

Please see the `prepare_data.sh` script. We get the Russian-Russian pairs of nodes with simple `grep` and build
a 3-dimensional array (source, target, relation) stored as a single sparse SciPy matrix.


## Citing

Please do not forget to cite the ConceptNet5 paper.
```
@article{speer2016conceptnet,
  title={Conceptnet 5.5: An open multilingual graph of general knowledge},
  author={Speer, Robyn and Chin, Joshua and Havasi, Catherine},
  journal={arXiv preprint arXiv:1612.03975},
  year={2016}
}
```

Citing the repository is not necessary, but greatly appreciated as well, if you use this work.

```
@misc{ruconceptnet2020alekseev,
  title     = {{alexeyev/RuConceptNet: /ru/ConceptNet5.7 Python wrapper }},
  year      = {2020},
  url       = {https://github.com/alexeyev/RuConceptNet},
  language  = {english}
}
```

## License

The code is released under the MIT license (please see the `LICENSE` file).

This work includes a subset data from ConceptNet 5, which was compiled by the
Commonsense Computing Initiative. ConceptNet 5 is freely available under
the Creative Commons Attribution-ShareAlike license (CC BY SA 3.0) from
http://conceptnet.io.

The included data was created by contributors to Commonsense Computing
projects, contributors to Wikimedia projects, DBPedia, OpenCyc, Games
with a Purpose, Princeton University's WordNet, Francis Bond's Open
Multilingual WordNet, and Jim Breen's JMDict.

The complete data in ConceptNet is available under the Creative Commons Attribution-ShareAlike 4.0 license.

For more details, please see ["Copying and sharing ConceptNet"](https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet).
