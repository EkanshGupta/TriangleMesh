# TriangleMesh
Combined code from MeshCNN & DGCNN for triangle mesh classification.

## Usage
- Run `get_data.sh`.
    - Or, manually download and extract the data from [here](https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz) to the `data/` folder (present in the gitignore)
- Create a config YAML file with all the fields present in the sample provided under `configs/`.
    - All instructions needed for filling out each field are present in the sample YAML file.
    - Note that the mentioned "default" values don't make the YAML fields optional. They are given simply to start you off with some good values, but you will need to fill them in the YAML nevertheless.
- Training:
```
python main.py train CONFIG_YAML_PATH
```
- Testing (i.e. eval on test set):
```
python main.py test CONFIG_YAML_PATH
```