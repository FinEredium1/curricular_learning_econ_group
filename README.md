# Fardin's Group 6 Results

INDPRO forecasting experiments using **Group 6: Interest and Exchange Rates** from FRED-MD.

Branch: `Fardin_group6_results`

## Starting point: Jayden's shared ZIP

These experiments use Jayden's shared `fredmd-team-setup-seed1.zip`. His related implementation and output-and-income study are available on the [output-and-income-jayden branch](https://github.com/FinEredium1/curricular_learning_econ_group/tree/output-and-income-jayden).

The ZIP supplied the common training code, frozen data and batch manifests, saved initial model weights, and pinned dependencies. I used that setup in Google Colab to run the Group 6 experiments and collect the results below.

Group 6 contains **22 auxiliary series**, covering interest rates, spreads, and exchange rates. **INDPRO stays in the training pool** for every condition. Each auxiliary example predicts its own series using the shared model; validation and testing measure INDPRO forecasting performance.

## Included files

| File or folder | Contents |
|---|---|
| [group6_results_seed1_20261031T203006Z_154216/](group6_results_seed1_20261031T203006Z_154216/) | Exported Group 6 run outputs for all 12 model/order combinations. |
| [FREDMD_Group6_All_Results.png](FREDMD_Group6_All_Results.png) | Comparison graph showing test MSE and MAE for all 12 runs, plus the zero-growth baseline. |
| [FREDMD_Group6_Colab_Setup.ipynb](FREDMD_Group6_Colab_Setup.ipynb) | Jupyter notebook used in Colab to load the shared ZIP, set up the environment, verify the inputs, run smoke checks, and launch Group 6 experiments. |

Inside the results export, individual runs follow `group6/<order>/<model>/seed1/`. Each run's `summary.json` records its actual model, ordering, update count, timing, and final test metrics.

## Experiment setup

| Setting | Value |
|---|---|
| Data | September 2026 FRED-MD snapshot |
| Evaluation target | INDPRO |
| Models | Linear, MLP, PatchTST |
| Forecasting task | 24 monthly observations predict the next 3 months |
| Training / validation / testing | 1960–1999 / 2000–2009 / 2010–2019 |
| Scaling | Per-series statistics fitted on training data only |
| Batch size | 32 |
| Training budget per run | 300 INDPRO batches + 500 auxiliary batches = 800 optimizer updates |
| Optimizer | Adam, learning rate 0.001 |
| Initialization | Same saved initial weights within each model; fresh optimizer for each run |
| Seed and backend | Seed 1; CPU with one PyTorch thread |
| Final evaluation | Final checkpoint; 118 INDPRO test windows |

The four orders use the same batches and exposure counts. Only their presentation order changes. Months within each forecasting example remain chronological.

| Order | Schedule |
|---|---|
| `random` | Shuffle all 800 target and auxiliary batches together. |
| `auxiliary_first` | 500 auxiliary batches, followed by 300 INDPRO batches. |
| `target_first` | 300 INDPRO batches, followed by 500 auxiliary batches. |
| `interleaved` | Repeat `T, A, A, T, A, A, T, A` 100 times, where T is INDPRO and A is auxiliary. |

Three models × four orders gives **12 completed research runs**. The initial smoke checks are separate setup checks.

## Results

![Group 6 test MSE and MAE for all 12 runs](FREDMD_Group6_All_Results.png)

Lower error is better. The table uses final test metrics **after reversing standardization**, aggregated over all 118 test windows and all three forecast horizons. MAE is in INDPRO growth units; MSE is in squared growth units.

| Model | Training order | Test MSE | Test MAE | Training time (s) |
|---|---|---:|---:|---:|
| Linear | Random | 0.2368 | 0.4066 | 2.735 |
| Linear | Auxiliary first | 0.2511 | 0.4187 | 2.454 |
| Linear | Target first | 0.2388 | 0.4046 | 2.760 |
| Linear | Interleaved | 0.2389 | 0.4070 | 2.590 |
| MLP | Random | **0.2266** | **0.3974** | 2.736 |
| MLP | Auxiliary first | 0.2486 | 0.4163 | 2.778 |
| MLP | Target first | 0.2341 | 0.3988 | 3.270 |
| MLP | Interleaved | 0.2311 | 0.4007 | 3.668 |
| PatchTST | Random | 0.2875 | 0.4450 | 12.494 |
| PatchTST | Auxiliary first | 0.2754 | 0.4313 | 12.132 |
| PatchTST | Target first | 0.2556 | 0.4115 | 12.936 |
| PatchTST | Interleaved | 0.2500 | 0.4185 | 14.736 |
| Predict zero growth | Baseline | 0.2373 | 0.3940 | N/A |

Training time measures optimizer-step computation after batch fetching. It excludes validation, testing, and plotting, and is not total notebook runtime.

### Findings

- **MLP with random ordering** had the lowest MSE and MAE among the 12 trained-model conditions.
- By MSE, random ordering worked best for Linear and MLP, while interleaving worked best for PatchTST. The best order therefore depended on the model.
- The **zero-growth baseline** still had lower MAE than every trained model. This baseline always predicts zero INDPRO growth in the original growth units.

These are preliminary results from one seed. Comparisons against the team's INDPRO-only controls at 300 and 800 updates are still needed to judge whether adding Group 6 data helps. These runs alone compare the specified schedules and do not establish a general benefit from auxiliary training or real-time forecasting performance.

## Using the notebook

Open `FREDMD_Group6_Colab_Setup.ipynb` in Google Colab and provide the original `fredmd-team-setup-seed1.zip` when prompted. Run the setup and verification cells before launching research conditions. The notebook creates an isolated study environment and backs up outputs to Google Drive.

Use the model and ordering recorded in each saved run's `summary.json` when aggregating results.
