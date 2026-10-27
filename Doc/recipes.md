# Paper Recipes

One row per experiment in the paper. Each recipe in `Config/experiment/<dataset>/` carries
the hyperparameters of the Hugging Face checkpoint in the same row. *Paper* is the
number in Tab. II (main comparison) or Tab. III (proprioceptive input); *Checkpoint* is
the released checkpoint's score on the test split. Both are AVE (m/s) / ATE (m).

`ssio_pgd` fine-tunes an `ssio` run: replace `<SSIO_RUN>` with that run's output
directory. Evaluate any run with
`python evaluate.py <RUN_DIR> --checkpoint best_model.pt --data Config/dataset/<Test split>.yaml`.

> [!NOTE]
>
> The released "Ours (w/ PGD)" checkpoints do not match the paper's PGD row; both are
> listed below.

## TLIO

| Paper row | Recipe | Paper | Checkpoint | Hugging Face run | Test split | Command |
|---|---|---|---|---|---|---|
| Tab. II AirIO† | `airio` | 0.084 / 2.08 | 0.0840 / 2.08 | [supervised/8pr646j5](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/8pr646j5) | `TLIO_Test` | `python train_ssio.py +experiment=tlio/airio` |
| Tab. II AirIO | `airio` + `network=airio` | 0.081 / 1.89 | – | [supervised/q1k41m3r](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/q1k41m3r) | `TLIO_Test` | `python train_ssio.py +experiment=tlio/airio network=airio` |
| Tab. II Ours | `ssio` | 0.104 / 3.04 | 0.1038 / 3.04 | [stage1/ev7hukzs](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/ev7hukzs) | `TLIO_Test` | `python train_ssio.py +experiment=tlio/ssio` |
| Tab. II Ours (w/ PGD) | `ssio_pgd` | 0.100 / 3.01 | 0.1007 / 2.99 | [stage2b/udfvlewc](https://huggingface.co/Open-IO/SSIO/tree/main/stage2b/udfvlewc) | `TLIO_Test` | `python train_pgd.py +experiment=tlio/ssio_pgd train.resume_from=file:<SSIO_RUN>/best_model.pt` |

## EuRoC

| Paper row | Recipe | Paper | Checkpoint | Hugging Face run | Test split | Command |
|---|---|---|---|---|---|---|
| Tab. II AirIO† | `airio` | 0.344 / 3.72 | 0.3444 / 3.72 | [supervised/ncntl12t](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/ncntl12t) | `EuRoC_Test` | `python train_ssio.py +experiment=euroc/airio` |
| Tab. II AirIO | `airio` + `network=airio` | 0.335 / 3.72 | – | [supervised/n50vs978](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/n50vs978) | `EuRoC_Test` | `python train_ssio.py +experiment=euroc/airio network=airio` |
| Tab. II Ours | `ssio` | 0.423 / 9.03 | 0.4227 / 9.03 | [stage1/189uubgg](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/189uubgg) | `EuRoC_Test` | `python train_ssio.py +experiment=euroc/ssio` |
| Tab. II Ours (w/ PGD) | `ssio_pgd` | 0.401 / 10.23 | 0.4031 / 9.93 | [stage2b/u0ej57z3](https://huggingface.co/Open-IO/SSIO/tree/main/stage2b/u0ej57z3) | `EuRoC_Test` | `python train_pgd.py +experiment=euroc/ssio_pgd train.resume_from=file:<SSIO_RUN>/best_model.pt` |

## Blackbird

| Paper row | Recipe | Paper | Checkpoint | Hugging Face run | Test split | Command |
|---|---|---|---|---|---|---|
| Tab. II AirIO† | `airio` | 0.362 / 6.39 | 0.3621 / 6.39 | [supervised/dvilntqz](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/dvilntqz) | `BlackBird_Test` | `python train_ssio.py +experiment=blackbird/airio` |
| Tab. II AirIO | `airio` + `network=airio` | 0.318 / 6.10 | – | [supervised/4ktbuoar](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/4ktbuoar) | `BlackBird_Test` | `python train_ssio.py +experiment=blackbird/airio network=airio` |
| Tab. II Ours | `ssio` | 0.475 / 18.39 | 0.4748 / 18.39 | [stage1/fpza91ml](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/fpza91ml) | `BlackBird_Test` | `python train_ssio.py +experiment=blackbird/ssio` |
| Tab. II Ours (w/ PGD) | `ssio_pgd` | 0.430 / 17.93 | 0.4290 / 17.73 | [stage2b/mks34gjg](https://huggingface.co/Open-IO/SSIO/tree/main/stage2b/mks34gjg) | `BlackBird_Test` | `python train_pgd.py +experiment=blackbird/ssio_pgd train.resume_from=file:<SSIO_RUN>/best_model.pt` |

## MIPO

| Paper row | Recipe | Paper | Checkpoint | Hugging Face run | Test split | Command |
|---|---|---|---|---|---|---|
| Tab. II AirIO† | `airio` | 0.121 / 0.31 | 0.1209 / 0.31 | [supervised/huamd5dg](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/huamd5dg) | `MIPO_Test` | `python train_ssio.py +experiment=mipo/airio` |
| Tab. II AirIO | `airio` + `network=airio` | 0.120 / 0.31 | – | [supervised/vqbya7lg](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/vqbya7lg) | `MIPO_Test` | `python train_ssio.py +experiment=mipo/airio network=airio` |
| Tab. II Ours | `ssio` | 0.182 / 1.63 | 0.1821 / 1.63 | [stage1/lwr3ayev](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/lwr3ayev) | `MIPO_Test` | `python train_ssio.py +experiment=mipo/ssio` |
| Tab. II Ours (w/ PGD) | `ssio_pgd` | 0.166 / 1.64 | 0.1692 / 1.58 | [stage2b/6059c2fr](https://huggingface.co/Open-IO/SSIO/tree/main/stage2b/6059c2fr) | `MIPO_Test` | `python train_pgd.py +experiment=mipo/ssio_pgd train.resume_from=file:<SSIO_RUN>/best_model.pt` |
| Tab. III AirIO† + *Proprio* | `airio_proprio` | 0.116 / 0.19 | 0.1164 / 0.19 | [supervised/4ch9v4hx](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/4ch9v4hx) | `MIPO_Test` | `python train_ssio.py +experiment=mipo/airio_proprio` |
| Tab. III Ours + *Proprio* | `ssio_proprio` | 0.146 / 0.82 | 0.1455 / 0.82 | [stage1/zueppy3z](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/zueppy3z) | `MIPO_Test` | `python train_ssio.py +experiment=mipo/ssio_proprio` |

## GrandTour

| Paper row | Recipe | Paper | Checkpoint | Hugging Face run | Test split | Command |
|---|---|---|---|---|---|---|
| Tab. II AirIO† | `airio` | 0.070 / 3.38 | 0.0702 / 3.38 | [supervised/ghgafmbs](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/ghgafmbs) | `ANYmal_Test` | `python train_ssio.py +experiment=grandtour/airio` |
| Tab. II AirIO | `airio` + `network=airio` | 0.068 / 3.19 | – | [supervised/oyor793p](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/oyor793p) | `ANYmal_Test` | `python train_ssio.py +experiment=grandtour/airio network=airio` |
| Tab. II Ours | `ssio` | 0.096 / 4.37 | 0.0958 / 4.37 | [stage1/3dqmg13r](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/3dqmg13r) | `ANYmal_Test` | `python train_ssio.py +experiment=grandtour/ssio` |
| Tab. II Ours (w/ PGD) | `ssio_pgd` | 0.095 / 4.34 | 0.0958 / 4.37 | [stage2b/gp86b7p6](https://huggingface.co/Open-IO/SSIO/tree/main/stage2b/gp86b7p6) | `ANYmal_Test` | `python train_pgd.py +experiment=grandtour/ssio_pgd train.resume_from=file:<SSIO_RUN>/best_model.pt` |
| Tab. III AirIO† + *Proprio* | `airio_proprio` | 0.073 / 3.62 | 0.0725 / 3.62 | [supervised/cdw3gb3x](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/cdw3gb3x) | `ANYmal_Test` | `python train_ssio.py +experiment=grandtour/airio_proprio` |
| Tab. III Ours + *Proprio* | `ssio_proprio` | 0.091 / 3.66 | 0.0907 / 3.66 | [stage1/cl7pny4l](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/cl7pny4l) | `ANYmal_Test` | `python train_ssio.py +experiment=grandtour/ssio_proprio` |

## Humanoid

| Paper row | Recipe | Paper | Checkpoint | Hugging Face run | Test split | Command |
|---|---|---|---|---|---|---|
| Tab. II AirIO† | `airio` | 0.096 / 1.40 | 0.0959 / 1.40 | [supervised/0102al6c](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/0102al6c) | `HumanoidOdom_Test` | `python train_ssio.py +experiment=humanoid/airio` |
| Tab. II AirIO | `airio` + `network=airio` | 0.087 / 1.26 | – | [supervised/c3qimksm](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/c3qimksm) | `HumanoidOdom_Test` | `python train_ssio.py +experiment=humanoid/airio network=airio` |
| Tab. II Ours | `ssio` | 0.167 / 4.36 | 0.1669 / 4.36 | [stage1/rt7jor4e](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/rt7jor4e) | `HumanoidOdom_Test` | `python train_ssio.py +experiment=humanoid/ssio` |
| Tab. II Ours (w/ PGD) | `ssio_pgd` | 0.148 / 4.27 | 0.1549 / 4.38 | [stage2b/r0vtbp2o](https://huggingface.co/Open-IO/SSIO/tree/main/stage2b/r0vtbp2o) | `HumanoidOdom_Test` | `python train_pgd.py +experiment=humanoid/ssio_pgd train.resume_from=file:<SSIO_RUN>/best_model.pt` |
| Tab. III AirIO† + *Proprio* | `airio_proprio` | 0.095 / 2.35 | 0.0953 / 2.35 | [supervised/lo9hemti](https://huggingface.co/Open-IO/SSIO/tree/main/supervised/lo9hemti) | `HumanoidOdom_Joints_Test` | `python train_ssio.py +experiment=humanoid/airio_proprio` |
| Tab. III Ours + *Proprio* | `ssio_proprio` | 0.161 / 3.20 | 0.1605 / 3.20 | [stage1/9gxajylc](https://huggingface.co/Open-IO/SSIO/tree/main/stage1/9gxajylc) | `HumanoidOdom_Joints_Test` | `python train_ssio.py +experiment=humanoid/ssio_proprio` |
