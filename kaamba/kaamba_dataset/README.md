---
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train/*"
  - split: test
    path: "data/test/*"
  - split: val
    path: "data/val/*"
    
- config_name: additional_data
  data_files: "additional_data.csv"
---