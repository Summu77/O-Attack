# O-Attack

**One Attack to Fool Them All: Highly Transferable Black-Box Adversarial Attacks on Frontier MLLMs**

Sen Nie · Jie Zhang · Zhongqi Wang · Shiguang Shan · Xilin Chen

Institute of Computing Technology, Chinese Academy of Sciences & University of Chinese Academy of Sciences

[Project page](https://summu77.github.io/O-Attack/) · [Paper](docs/assets/o-attack-paper.pdf)

O-Attack exploits a broad, high-level, cross-modally aligned semantic space within surrogate models through semantic space anchoring, progressive sampling, and semantic consensus optimization. Experiments span 10 frontier commercial MLLMs and 14 widely used MLLMs, with six state-of-the-art baselines.

The project website includes the method overview, interactive experimental results, qualitative examples, and practical safety implications. Implementation and configuration release updates will be provided in this repository.

## Project website

The website is a static site in `docs/`, with no build dependencies. To preview locally:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8765/ in your browser.

To publish, open **Settings → Pages → Deploy from a branch**, select **main** and **/docs**, and click **Save**. See the [Chinese configuration guide](GITHUB_PAGES_配置说明.md) for detailed instructions.
