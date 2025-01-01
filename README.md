# Clustering Exploration

-----

## Installation

- [Install Hatch](https://hatch.pypa.io/latest/install/) (this is a python environment manager)
- Clone repository
- Run `hatch shell` to enter the virtual environment
- Put `collected_splats.csv` and `gt.png` in the `src/clustering_exploration/data/` directory
  - These were generated using [this fork](https://github.com/PLSE-Splats/diff-gaussian-rasterization/tree/extract-all-splats) of `diff-gaussian-rasterization`
- Run through the notebooks in `src/cluster_exploration`.

## Testing
- The test setup can be found in `tests` folder
- Before running any program, modify the `config.json` with correct file path.
- `dataset_init.py` will create csv files for views from given model.
  - Usage:
```python dataset_init.py <path-to-config>```
- `test_suite.py` will compute images and corresponding statistics from clustering algorithm.
  - Usage:
```python test_suite.py <path-to-config>```

## License

`clustering-exploration` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
