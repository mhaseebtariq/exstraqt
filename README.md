# ExSTraQt
[**Ex**tract **S**uspicious **Tra**nsactions with **Q**uasi-**t**emporal (Graph Modeling)]
A supervised machine learning framework for identifying money laundering transactions in bank data.

## Setup:
* Important NOTES:
  * `*_prepare_input_*.ipynb` notebooks work best with Python 3.9.19
  * The rest of the notebooks work best with Python 3.11.8
  * Use `1__prepare_input_wrapper` to generate inputs for all the datasets at once
  * Use `2__model_wrapper.ipynb` to generate outputs with different seeds
* Make sure that all the relevant open-source datasets are downloaded to `./data/`
  * https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network
  * https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
* Activate the relevant Python or Conda environment
* Install requirements: `pip install -r requirements.txt`
* Install `prepare-data` requirements (in a different environment): `pip install -r requirements-prepare-data.txt`
* Start Jupyter server: `jupyter lab`
* Please execute the notebooks, in order `1_<name>.ipynb`, `2_<name>.ipynb`, ...
