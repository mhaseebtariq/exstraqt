# ExSTraQt
**Ex**tract **S**uspicious **Tra**nsactions with **Q**uasi-**t**emporal (Graph Modeling)

A supervised machine learning framework for identifying money laundering transactions in bank data.

## Setup:
* Important NOTES:
  * `1_prepare_input.ipynb` works best with Python 3.9.19
  * The rest of the notebooks work best with Python 3.11.8
* Make sure that all the relevant open-source datasets are downloaded
  * https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network
  * https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
* Activate the relevant Python or Conda environment
* Install requirements: `pip install -r requirements.txt`
* Install `prepare-data` requirements (in a different environment): `pip install -r requirements-prepare-data.txt`
* Start Jupyter server: `jupyter lab`
* Please execute the notebooks, in order `1_<name>.ipynb`, `2_<name>.ipynb`, ...
