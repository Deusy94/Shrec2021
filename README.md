<h1 align="center">Shrec 2021 submission</h1>

*A. D’Eusanio, A. Simoni, S. Pini, G. Borghi, R. Vezzani, R. Cucchiara*
 
We propose a transformer-based architecture for the dynamic hand gesture recognition task, using the features provided 
by the Leap Motion camera.

General script and test parameters:
```
python main.py \
--hypes /path/to/test_realtime.json \
--phase test_realtime \
--resume /path/to/checkpoints.pth
```

We propose 3 different settings, you can test them with the following commands:
- `python main.py --hypes ./hyperparameters/shrec/test_realtime_a.json`
- `python main.py --hypes ./hyperparameters/shrec/test_realtime_b.json`
- `python main.py --hypes ./hyperparameters/shrec/test_realtime_c.json`

Results are saved in the `outputs` folder.

Inside the `checkpoints` folder (download it from [here](https://drive.google.com/drive/folders/1WOuSuqwmB23dtkl98cm-lyfRN-yIp90u)) you can find the 3 files related to our 3 submissions:
- best_full_dataset.pth
- best_full_dataset_norm.pth
- best_full_dataset_moredata.pth

Additional information about the config file:
- "n_features": [20, 13] -> if resuming submission a (default normalization) or b (normalization with hand dimension)
- "n_features": [20, 32] -> if resuming submission c (default normalization + joints distances)
- "norm_hand" -> true if resuming submission b (hand dimension normalization) else false
- "distances" -> true if resuming submission c (joint distances as additional features) else false
