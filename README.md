Minigrid RL
====

## Ollama

Install "nvidia-container-toolkit".

Run server:
* `docker run --rm -d -e OLLAMA_FLASH_ATTENTION=1 -e OLLAMA_KV_CACHE_TYPE=q4_0 --runtime=nvidia -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`

Pull model:
* `docker exec -it ollama ollama pull <model>`

Run interactive session:
* `docker exec -it ollama ollama run --verbose <model>`

List of [models](https://ollama.com/library).

## Trening parameters

| Model | Epoch | LR init | LR final |
|:-----:|:-----:|:-------:|:--------:|
| gtg0  |   6   |  1e-3   |   3e-5   |
| gtg1  |   4   |  3e-4   |   3e-6   |
| pkp0  |   9   |  1e-3   |   3e-5   |
| pkp1  |   9   |  3e-4   |   3e-6   |
| pkp2  |   9   |  3e-4   |   3e-6   |
| pkp3  |   9   |  3e-4   |   3e-6   |
| pkp4  |   6   |  2e-4   |   3e-6   |
| pkpc1 |   4   |  3e-4   |   3e-6   |
| pkpc2 |   4   |  3e-4   |   3e-6   |
| pkpc3 |   4   |  3e-4   |   3e-6   |
| gto0  |   7   |  1e-3   |   3e-5   |
| gto1  |   7   |  3e-4   |   3e-6   |
| gto2  |   4   |  3e-4   |   3e-6   |
| tgl0  |   4   |  1e-3   |   3e-5   |
| tgl1  |   4   |  3e-4   |   3e-6   |
| tgl2  |   4   |  3e-4   |   3e-6   |
| tgl3  |   4   |  3e-4   |   3e-6   |
| tglc1 |   4   |  3e-4   |   3e-6   |
| tglc2 |   4   |  3e-4   |   3e-6   |
| all0  |   6   |  1e-3   |   3e-5   |
| all1  |   6   |  3e-4   |   3e-6   |
| all2  |   6   |  3e-4   |   3e-6   |
| all3  |   4   |  3e-4   |   3e-6   |
| all4  |   4   |  3e-4   |   3e-6   |
| all5  |   4   |  3e-4   |   3e-6   |
| all6  |   4   |  2e-4   |   3e-6   |
| nlm0  |   6   |  1e-3   |   3e-5   |
| nlm1  |   6   |  1e-3   |   3e-5   |
| nlm2  |   4   |  2e-4   |   3e-6   |


## Benchmark (1k ep)

### PPO
| Model | GTG | GTO | PKP | TGL | ALL |
|:-----:|:---:|:---:|:---:|:---:|:---:|
|  GTG  | 86% |  0% |  0% |  0% | 19% |
|  GTO  |  0% | 72% |  0% |  0% | 17% |
|  PKP  |  0% |  0% | 57% |  0% | 26% |
|  PKPC |  0% |  0% | 68% |  0% | 32% |  # fine tune all model
|  TGL  |  0% |  0% |  0% | 47% | 27% |
|  TGLC |  0% |  0% |  0% | 65% | 47% |  # fine tune all model
|  ALL  | 75% | 65% | 59% | 58% | 65% |

### PPO vs Distillation vs MOE
| Problem | PPO | DIS | CON | MOE |
|:-------:|:---:|:---:|:---:|:---:|
|   GTG   | 86% | 86% | 86% | xx% |
|   GTO   | 72% | 64% | 73% | xx% |
|   PKP   | 57% | 47% | 66% | xx% |
|   TGL   | 65% | 37% | 64% | xx% |
|   ALL   | 65% | 56% | 67% | 72% |

### LLM
| Model | Result |
|:-----:|:------:|
|  PPO  |  33%   |
|  MOE  |  57%   |
|  DIS  |  55%   |
|  NLM  |  43%   |

|      NLM     |     DIS    |    PPO    |     MOE    |
|:------------:|:----------:|:---------:|:----------:|
| 0            | 0.7322314  | 0         | 0.74710745 |
| 0.9553719163 | 0.9479339  | 0.9486359 | 0.9469437  |
| 0            | 0.7247934  | 0         | 0.7237845  |
| 0            | 0          | 0         | 0          |
| 0.7247933745 | 0.7563426  | 0.7127631 | 0.7157412  |
| 0.8289256096 | 0.7523325  | 0.7322314 | 0.7214331  |
| 0            | 0.57603306 | 0         | 0          |
| 0.8661156893 | 0          | 0         | 0.87412578 |
| 0            | 0          | 0         | 0          |
| 0.9553719163 | 0.9628099  | 0.9436017 | 0.9518792  |
