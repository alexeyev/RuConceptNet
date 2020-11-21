# RussianConceptNet
ConceptNet 5.7, Russian part extraction scripts + fast access to relations

### Preparations

Please see the `prepare_data.sh` script. We get the Russian-Russian pairs of nodes with simple `grep` and build
a 3-dimensional array (source, target, relation) stored as a single sparse SciPy matrix.

### Usage

```pythonstub
from conceptnet import ConceptNet

cn = ConceptNet()
print(cn.get_targets("красота"))
print(cn.get_sources("красота"))
print(cn.check_pair("человек", "зверь"))
print(cn.check_pair("зверь", "человек"))
```

### 

## Citing

ConceptNet5 paper:
```
Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge." In proceedings of AAAI 31.
```

Citing the repository is greatly appreciated as well, if you use this work.

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