Reference code for the **ucb_ego** submission to the [EgoExo4D BodyPose Challenge](https://eval.ai/web/challenges/challenge-page/2245/overview).

To install, this repository can be cloned and set up via pip:

```sh
git clone git@github.com:brentyi/ucb_ego.git
cd ucb_ego
pip install -e .
```

Development was done in Python 3.10.

You can find the checkpoint file (`experiments.zip`, ~526MB), as well as the 4 page validation report that we submitted to the challenge in [this](https://drive.google.com/drive/u/1/folders/1yWRFJO7kZf2rKJ9WggUc_OyvCHZRKlje) Google Drive folder.

We've refactored our pipeline into six individual scripts, which are ordered as follows:

- **`0_compute_floor_heights.py`**
  - Use RANSAC to estimate floor heights for EgoExo4D's EgoPose training trajectories.
- **`1_visualize_floor_heights.py`**
  - Qualitative verification for the floor heights.
- **`2_train_adapter.py`**
  - Train our adapter network, which predicts floor heights and take type for test trajectories.
- **`3_sample_from_test_set.py`**
  - Sample trajectories from diffusion model, conditioned on test set device poses.
- **`4_make_egopose_submission_json.py`**
  - Convert sampling outputs to JSON format needed for EgoExo4D challenge leaderboard.
- **`5_multiplex_outputs.py`**
  - Use take type to multiplex outputs from our JSON file and the
    competition-provided baseline.

Each script is set up as a CLI interface. To view helptext and inputs, you can run
`python {script_name}.py --help`.
