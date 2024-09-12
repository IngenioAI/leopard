## Privacy Risk Assessment tool for Machine Learning Models

### For Tensorflow (.h5) and Pytorch (.pt) Models

`Must have Python version >= 3.9`  and  `install -r requirements.txt`

Arguments for running  `main.py` for attacks:

    --model_path  | MODEL_PATH  Absolute path where pretrained model is saved                            
    --attack      | ATTACK      Attack type: "custom" | "lira" | "population" | "reference" | "shadow"
    --n_class     | N_CLASS     Number of classes of target model dataset, default is 10 (for Cifar10). Pass 100    
                                for Cifar100 data and target model trained with this data.

### `Important! The pretrained model should be saved as a whole model object, not as state_dict format which requires model initialization.`

If no target model exist then train it first with or without Differential Privacy:

    $ python main.py --model_path /path/to/model --train True 

For `lira attacks` you can change the config file in `attacks/config.py`

    aconf = {
        'lr': 0.02,
        'batch_size': 128,
        'epochs': 2,
        'n_shadows': 2,
        'shpath': './attacks/shadows'
    }

For `population` and `reference` metric attacks same config file as above but:

    priv_meter = {        
        'num_train_points': 10000,
        'num_test_points': 10000,
        'epochs': 10,
        'batch_size': 64,
        'num_population_points': 10000,
        'fpr_tolerance_list': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'input_shape': (224, 224, 3),
        'ref_models': './attacks/shadows/',
        'torch_loss': torch.nn.CrossEntropyLoss(reduction='none'),
        'tf_loss': tf.keras.losses.CategoricalCrossentropy()
    }

parameters can be updated.